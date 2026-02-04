"""
FreeFuse Mask Applicator Node

Applies Phase 1 masks to LoRA-loaded model for Phase 2 generation.
This is the bridge between Phase 1 (mask collection) and Phase 2 (generation).

Uses Plan A: FreeFuseBypassForwardHook to intercept LoRA outputs and apply
spatial masks at the h(x) output level.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Optional, List, Tuple

import comfy.model_patcher
import comfy.weight_adapter
from comfy.weight_adapter.base import WeightAdapterBase, WeightAdapterTrainBase

from ..freefuse_core.freefuse_bypass import (
    FreeFuseBypassForwardHook,
    FreeFuseBypassInjectionManager,
)


class FreeFuseMaskApplicator:
    """
    Apply FreeFuse masks to model with loaded LoRAs.
    
    This node takes:
    - Model with LoRAs loaded in bypass mode
    - Masks from Phase 1 sampler
    - FreeFuse data with adapter info
    
    And outputs a model with masked LoRA application.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "masks": ("FREEFUSE_MASKS",),
                "freefuse_data": ("FREEFUSE_DATA",),
            },
            "optional": {
                "enable_token_masking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Zero out LoRA at other concepts' token positions"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Optional latent for size reference"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_masks"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Apply Phase 1 masks to LoRAs for Phase 2 generation.
    
Connect this between your LoRA loaders and KSampler.
The masks from FreeFusePhase1Sampler control where each LoRA affects the image."""
    
    def apply_masks(
        self,
        model,
        masks,
        freefuse_data,
        enable_token_masking=True,
        latent=None,
    ):
        # Extract data
        mask_dict = masks.get("masks", {})
        adapters = freefuse_data.get("adapters", [])
        token_pos_maps = freefuse_data.get("token_pos_maps", {})
        
        if not mask_dict:
            print("[FreeFuse] Warning: No masks provided, returning model unchanged")
            return (model,)
        
        if not adapters:
            print("[FreeFuse] Warning: No adapters registered, returning model unchanged")
            return (model,)
        
        # Determine latent size from masks or latent input
        latent_size = self._get_latent_size(mask_dict, latent)
        if latent_size is None:
            print("[FreeFuse] Warning: Could not determine latent size")
            return (model,)
        
        # Clone the model
        model_clone = model.clone()
        
        # Get adapter name to mask mapping
        adapter_mask_map = {}
        for adapter_info in adapters:
            name = adapter_info.get("name")
            if name and name in mask_dict:
                adapter_mask_map[name] = {
                    "mask": mask_dict[name],
                    "info": adapter_info,
                }
            elif name:
                print(f"[FreeFuse] Warning: No mask found for adapter '{name}'")
        
        if not adapter_mask_map:
            print("[FreeFuse] Warning: No adapter-mask mappings, returning model unchanged")
            return (model,)
        
        # Apply masks using the hook system
        self._apply_masks_to_model(
            model_clone,
            adapter_mask_map,
            latent_size,
            token_pos_maps if enable_token_masking else None,
        )
        
        print(f"[FreeFuse] Applied masks to {len(adapter_mask_map)} adapters")
        print(f"[FreeFuse] Latent size: {latent_size}")
        
        return (model_clone,)
    
    def _get_latent_size(
        self,
        mask_dict: Dict[str, torch.Tensor],
        latent: Optional[Dict],
    ) -> Optional[Tuple[int, int]]:
        """Determine latent size from masks or latent input."""
        # Try from latent input
        if latent is not None and "samples" in latent:
            samples = latent["samples"]
            return (samples.shape[2], samples.shape[3])
        
        # Try from mask dimensions
        for mask in mask_dict.values():
            if mask.dim() == 2:
                return (mask.shape[0], mask.shape[1])
            elif mask.dim() == 3:
                return (mask.shape[1], mask.shape[2])
        
        return None
    
    def _apply_masks_to_model(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
        adapter_mask_map: Dict[str, Dict],
        latent_size: Tuple[int, int],
        token_pos_maps: Optional[Dict[str, List[int]]],
    ):
        """Apply masks to model's LoRA layers using FreeFuse bypass hooks.
        
        This uses Plan A: Create FreeFuseBypassForwardHook instances that wrap
        each LoRA layer and apply spatial masks to h(x) output before adding
        to base output.
        
        Formula: output = g(f(x) + mask * h(x))
        """
        # Create the FreeFuse bypass injection manager
        manager = FreeFuseBypassInjectionManager()
        
        # Collect all masks
        masks = {}
        for name, data in adapter_mask_map.items():
            masks[name] = data["mask"]
        
        # Get existing patches from the model
        patches = getattr(model_patcher, 'patches', {})
        hooks = getattr(model_patcher, 'hook_patches', {})
        
        logging.info(f"[FreeFuse] Model has {len(patches)} patches, {len(hooks)} hooks")
        
        # Find LoRA adapters and register them with FreeFuse manager
        adapter_found = False
        
        for key, patch_list in patches.items():
            for patch in patch_list:
                # patch format: (strength, adapter, ...) or other formats
                if not isinstance(patch, tuple) or len(patch) < 2:
                    continue
                
                strength = patch[0] if isinstance(patch[0], (int, float)) else 1.0
                adapter_data = patch[1]
                
                # Check if this is a bypass adapter
                if isinstance(adapter_data, (WeightAdapterBase, WeightAdapterTrainBase)):
                    # Determine which concept/adapter this belongs to
                    adapter_name = self._find_adapter_name_for_key(
                        key, adapter_mask_map, model_patcher
                    )
                    
                    if adapter_name and adapter_name in masks:
                        manager.add_adapter(
                            key,
                            adapter_data,
                            strength=float(strength),
                            adapter_name=adapter_name,
                        )
                        adapter_found = True
                        logging.debug(f"[FreeFuse] Registered adapter: {key} -> {adapter_name}")
        
        if not adapter_found:
            logging.warning("[FreeFuse] No bypass adapters found in model patches")
            logging.info("[FreeFuse] Falling back to model_options mask storage")
            self._fallback_mask_storage(model_patcher, masks, latent_size, token_pos_maps)
            return
        
        # Get the diffusion model
        diffusion_model = self._get_diffusion_model(model_patcher)
        if diffusion_model is None:
            logging.error("[FreeFuse] Could not get diffusion model")
            return
        
        # Create injections
        injections = manager.create_injections(diffusion_model)
        
        # Apply masks to the manager
        manager.set_masks(masks, latent_size)
        
        # Apply token position masking if enabled
        if token_pos_maps:
            manager.set_token_positions(token_pos_maps)
        
        # Determine txt_len for Flux models
        txt_len = self._get_txt_len(model_patcher)
        if txt_len:
            manager.set_txt_len(txt_len)
        
        # Register injections with the model patcher
        model_patcher.set_injections("freefuse_bypass", injections)
        
        # Store manager in model_options for access during generation
        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}
        
        model_patcher.model_options["transformer_options"]["freefuse_bypass_manager"] = manager
        model_patcher.model_options["transformer_options"]["freefuse_masks"] = masks
        model_patcher.model_options["transformer_options"]["freefuse_latent_size"] = latent_size
        
        logging.info(f"[FreeFuse] Successfully applied masks via bypass hooks")
        logging.info(f"[FreeFuse] Registered {len(manager.hooks)} FreeFuse hooks")
    
    def _find_adapter_name_for_key(
        self,
        key: str,
        adapter_mask_map: Dict[str, Dict],
        model_patcher: comfy.model_patcher.ModelPatcher,
    ) -> Optional[str]:
        """Find which adapter/concept a weight key belongs to.
        
        This matches weight keys to adapter names by checking the freefuse_data
        stored in model_options, or by inferring from the adapter_mask_map.
        """
        # Check if freefuse_data has adapter_keys mapping
        transformer_options = model_patcher.model_options.get("transformer_options", {})
        freefuse_data = transformer_options.get("freefuse_data", {})
        adapter_keys = freefuse_data.get("adapter_keys", {})
        
        # Search in adapter_keys
        for adapter_name, keys in adapter_keys.items():
            if key in keys:
                return adapter_name
        
        # Check if there's an adapter_key_to_name mapping
        key_to_name = freefuse_data.get("adapter_key_to_name", {})
        if key in key_to_name:
            return key_to_name[key]
        
        # If only one adapter in mask_map, use it
        adapter_names = list(adapter_mask_map.keys())
        if len(adapter_names) == 1:
            return adapter_names[0]
        
        # Can't determine - return None
        logging.debug(f"[FreeFuse] Could not determine adapter for key: {key}")
        return None
    
    def _get_diffusion_model(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
    ):
        """Get the actual diffusion model from the patcher."""
        model = model_patcher.model
        if hasattr(model, 'diffusion_model'):
            return model.diffusion_model
        elif hasattr(model, 'model'):
            if hasattr(model.model, 'diffusion_model'):
                return model.model.diffusion_model
            return model.model
        return model
    
    def _get_txt_len(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
    ) -> Optional[int]:
        """Get text sequence length for Flux models."""
        # Check if already stored
        transformer_options = model_patcher.model_options.get("transformer_options", {})
        if "txt_len" in transformer_options:
            return transformer_options["txt_len"]
        
        # Check freefuse_data
        freefuse_data = transformer_options.get("freefuse_data", {})
        if "txt_len" in freefuse_data:
            return freefuse_data["txt_len"]
        
        # Default for Flux T5 + CLIP
        return None
    
    def _fallback_mask_storage(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
        masks: Dict[str, torch.Tensor],
        latent_size: Tuple[int, int],
        token_pos_maps: Optional[Dict[str, List[int]]],
    ):
        """Fallback: Store masks in model_options for use by wrapper."""
        if "transformer_options" not in model_patcher.model_options:
            model_patcher.model_options["transformer_options"] = {}
        
        model_patcher.model_options["transformer_options"]["freefuse_masks"] = masks
        model_patcher.model_options["transformer_options"]["freefuse_latent_size"] = latent_size
        
        if token_pos_maps:
            model_patcher.model_options["transformer_options"]["freefuse_token_pos_maps"] = token_pos_maps
        
        # Set up the model wrapper for fallback mask application
        self._wrap_model_forward(model_patcher, masks, latent_size, token_pos_maps)
    
    def _wrap_model_forward(
        self,
        model_patcher: comfy.model_patcher.ModelPatcher,
        masks: Dict[str, torch.Tensor],
        latent_size: Tuple[int, int],
        token_pos_maps: Optional[Dict[str, List[int]]],
    ):
        """Wrap the model's forward pass to apply masks to LoRA outputs.
        
        ComfyUI's bypass mode applies LoRA as: output = base + lora_path
        We modify this to: output = base + mask * lora_path
        
        This is done via the sampler_cfg_function callback.
        """
        # Store configuration for the wrapper
        wrapper_config = {
            "masks": masks,
            "latent_size": latent_size,
            "token_pos_maps": token_pos_maps,
            "enabled": True,
        }
        
        model_patcher.model_options["transformer_options"]["freefuse_wrapper_config"] = wrapper_config
        
        # Use set_model_unet_function_wrapper for deeper control
        original_wrapper = model_patcher.model_options.get("model_function_wrapper")
        
        def freefuse_model_wrapper(model_function, params):
            """Wrapper that enables mask application during forward."""
            # Set up mask context
            x = params.get("input", params.get("x"))
            
            if x is not None and wrapper_config["enabled"]:
                # Store current batch info for mask application
                batch_size = x.shape[0]
                spatial_size = x.shape[2:]  # (H, W) or (T, H, W)
                
                wrapper_config["current_batch"] = batch_size
                wrapper_config["current_spatial"] = spatial_size
            
            # Call original wrapper if exists
            if original_wrapper is not None:
                return original_wrapper(model_function, params)
            else:
                return model_function(params["input"], params["timestep"], **params.get("c", {}))
        
        model_patcher.set_model_unet_function_wrapper(freefuse_model_wrapper)


class FreeFuseMaskDebug:
    """
    Debug node to inspect FreeFuse masks.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("FREEFUSE_MASKS",),
            },
            "optional": {
                "target_size": ("INT", {
                    "default": 512, "min": 64, "max": 2048,
                    "tooltip": "Target size for visualization"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "info")
    FUNCTION = "debug_masks"
    CATEGORY = "FreeFuse"
    
    DESCRIPTION = """Visualize FreeFuse masks for debugging."""
    
    def debug_masks(self, masks, target_size=512):
        mask_dict = masks.get("masks", {})
        sim_maps = masks.get("similarity_maps", {})
        
        # Build info string
        info_lines = ["FreeFuse Mask Debug:"]
        info_lines.append(f"  Masks: {len(mask_dict)}")
        for name, mask in mask_dict.items():
            info_lines.append(f"    {name}: shape={tuple(mask.shape)}, "
                            f"min={mask.min():.3f}, max={mask.max():.3f}")
        
        info_lines.append(f"  Similarity maps: {len(sim_maps)}")
        for name, sim in sim_maps.items():
            info_lines.append(f"    {name}: shape={tuple(sim.shape)}")
        
        info = "\n".join(info_lines)
        
        # Create visualization
        preview = self._create_debug_preview(mask_dict, target_size)
        
        return (preview, info)
    
    def _create_debug_preview(self, masks: Dict[str, torch.Tensor], target_size: int) -> torch.Tensor:
        """Create a multi-panel debug visualization."""
        if not masks:
            return torch.zeros(1, target_size, target_size, 3)
        
        colors = [
            (1.0, 0.3, 0.3),   # Red
            (0.3, 1.0, 0.3),   # Green
            (0.3, 0.3, 1.0),   # Blue
            (1.0, 1.0, 0.3),   # Yellow
            (1.0, 0.3, 1.0),   # Magenta
            (0.3, 1.0, 1.0),   # Cyan
            (0.5, 0.5, 0.5),   # Gray
        ]
        
        # Create combined visualization
        combined = torch.zeros(3, target_size, target_size)
        
        for i, (name, mask) in enumerate(masks.items()):
            color = colors[i % len(colors)]
            
            # Ensure mask is 2D
            if mask.dim() == 3:
                mask = mask[0]
            
            # Resize to target
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Add colored mask
            for c in range(3):
                combined[c] += mask_resized * color[c]
        
        # Normalize
        combined = combined.clamp(0, 1)
        
        # Convert to (B, H, W, C)
        preview = combined.permute(1, 2, 0).unsqueeze(0)
        
        return preview


# Export node mappings
NODE_CLASS_MAPPINGS = {
    "FreeFuseMaskApplicator": FreeFuseMaskApplicator,
    "FreeFuseMaskDebug": FreeFuseMaskDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseMaskApplicator": "FreeFuse Mask Applicator",
    "FreeFuseMaskDebug": "FreeFuse Mask Debug",
}
