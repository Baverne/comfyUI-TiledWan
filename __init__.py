from .TiledWanVideoVacePipe import TiledWanVideoVACEpipe
from .TiledWanPreprocessing import ImageToMask, ImageStatistics, MaskStatistics, InpaintCropImproved, InpaintStitchImproved



NODE_CLASS_MAPPINGS = {
    "TiledWanImageToMask": ImageToMask,
    "TiledWanImageStatistics": ImageStatistics,
    "TiledWanMaskStatistics": MaskStatistics,
    "TiledWanVideoVACEpipe": TiledWanVideoVACEpipe,
    "TiledWanInpaintCropImproved": InpaintCropImproved,
    "TiledWanInpaintStitchImproved": InpaintStitchImproved,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledWanImageToMask": "TiledWan Image To Mask",
    "TiledWanImageStatistics": "TiledWan Image Statistics",
    "TiledWanMaskStatistics": "TiledWan Mask Statistics",
    "TiledWanVideoVACEpipe": "TiledWan Video VACE Pipeline",
    "TiledWanInpaintCropImproved": "TiledWan Inpaint Crop Improved",
    "TiledWanInpaintStitchImproved": "TiledWan Inpaint Stitch Improved"
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']