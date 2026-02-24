"""FreeFuse transformer wrapper for HiDream i1.

The HiDream i1 integration uses the same control and masking semantics as
Z-Image's single-stream transformer wrapper.
"""

from src.models.freefuse_transformer_z_image import (
    ZImageTransformer2DModel as FreeFuseHiDreamI1Transformer2DModel,
    ZImageTransformerBlock as HiDreamI1TransformerBlock,
    SEQ_MULTI_OF,
)

__all__ = [
    "FreeFuseHiDreamI1Transformer2DModel",
    "HiDreamI1TransformerBlock",
    "SEQ_MULTI_OF",
]
