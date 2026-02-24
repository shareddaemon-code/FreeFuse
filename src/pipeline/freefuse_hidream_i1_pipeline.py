"""FreeFuse pipeline for HiDream i1.

This pipeline subclasses the Z-Image FreeFuse pipeline because HiDream i1
shares the same unified-sequence execution path in this repository.
"""

from src.pipeline.freefuse_z_image_pipeline import FreeFuseZImagePipeline


class FreeFuseHiDreamI1Pipeline(FreeFuseZImagePipeline):
    """FreeFuse two-phase pipeline entrypoint for HiDream i1."""

