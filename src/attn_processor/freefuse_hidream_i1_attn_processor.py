"""FreeFuse attention processor for HiDream i1.

HiDream i1 follows the same single-stream attention behavior as Z-Image in
FreeFuse, so this class reuses the Z-Image implementation while providing a
first-class architecture-specific entrypoint.
"""

from src.attn_processor.freefuse_z_image_attn_processor import FreeFuseZImageAttnProcessor


class FreeFuseHiDreamI1AttnProcessor(FreeFuseZImageAttnProcessor):
    """HiDream i1 FreeFuse attention processor (single-stream)."""

