"""
FreeFuse Mask Generation Utilities
"""

import torch
import torch.nn.functional as F


def balanced_argmax(stacked, iterations=50, balance_weight=0.1):
    """
    Stabilized balanced argmax algorithm.
    
    Iteratively adjusts biases to balance area allocation
    across all concepts.
    
    Args:
        stacked: (N, H, W) tensor of similarity maps
        iterations: Number of balancing iterations
        balance_weight: How aggressively to balance
        
    Returns:
        (N, H, W) tensor of binary masks
    """
    n_concepts, h, w = stacked.shape
    total_pixels = h * w
    target_count = total_pixels // n_concepts
    
    biases = torch.zeros(n_concepts, device=stacked.device, dtype=stacked.dtype)
    
    for _ in range(iterations):
        adjusted = stacked + biases.view(-1, 1, 1)
        assignment = adjusted.argmax(dim=0)
        
        counts = torch.bincount(assignment.flatten(), minlength=n_concepts)
        counts = counts[:n_concepts].float()  # Ensure correct length
        
        diff = counts - target_count
        biases = biases - balance_weight * diff / total_pixels
    
    # Final assignment
    adjusted = stacked + biases.view(-1, 1, 1)
    assignment = adjusted.argmax(dim=0)
    
    # Convert to per-concept masks
    masks = torch.zeros_like(stacked)
    for i in range(n_concepts):
        masks[i] = (assignment == i).float()
    
    return masks


def generate_masks(similarity_maps, include_background=True, method="balanced"):
    """
    Generate masks from similarity maps.
    
    Args:
        similarity_maps: Dict[name -> (H, W) tensor]
        include_background: Whether to add background class
        method: "balanced" or "argmax"
        
    Returns:
        Dict[name -> (H, W) mask tensor]
    """
    if not similarity_maps:
        return {}
    
    names = list(similarity_maps.keys())
    maps = [similarity_maps[n] for n in names]
    
    # Ensure same shape
    target_shape = maps[0].shape[-2:]
    device = maps[0].device
    dtype = maps[0].dtype
    
    maps_resized = []
    for m in maps:
        if m.shape[-2:] != target_shape:
            m = F.interpolate(
                m.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze()
        maps_resized.append(m.to(device=device, dtype=dtype))
    
    # Stack: (N, H, W)
    stacked = torch.stack(maps_resized, dim=0)
    
    # Add background if needed
    if include_background:
        bg = torch.zeros_like(stacked[0:1])
        stacked = torch.cat([stacked, bg], dim=0)
        names = names + ["_background_"]
    
    # Generate masks
    if method == "balanced":
        mask_tensor = balanced_argmax(stacked)
    else:
        # Simple argmax
        assignment = stacked.argmax(dim=0)
        mask_tensor = torch.zeros_like(stacked)
        for i in range(len(names)):
            mask_tensor[i] = (assignment == i).float()
    
    # Convert to dict
    result = {}
    for i, name in enumerate(names):
        result[name] = mask_tensor[i]
    
    return result


def resize_mask(mask, target_size):
    """Resize mask to target size."""
    if mask.shape[-2:] == target_size:
        return mask
    
    return F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=target_size,
        mode='nearest'
    ).squeeze()
