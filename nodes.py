import torch
import comfy.utils
import comfy.model_management
import node_helpers


class ImageToMask:
    """
    Convert an image to a mask by extracting a specific channel.
    Supports red, green, blue, and alpha channels with optional clamping and normalization.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (["red", "green", "blue", "alpha"],),
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
                "normalize_output": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled", 
                    "label_off": "disabled"
                }),
            },
        }

    CATEGORY = "TiledWan"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "image_to_mask"

    def image_to_mask(self, image, channel, clamp_output, normalize_output):
        """
        Convert image to mask by extracting specified channel.
        
        Args:
            image: Input image tensor [batch, height, width, channels]
            channel: Channel to extract ("red", "green", "blue", "alpha")
            clamp_output: Whether to clamp values between 0 and 1
            normalize_output: Whether to normalize values to 0-1 range
            
        Returns:
            tuple: Mask tensor [batch, height, width]
        """
        channels = ["red", "green", "blue", "alpha"]
        channel_index = channels.index(channel)
        
        # Handle case where alpha channel is requested but doesn't exist
        if channel_index == 3 and image.shape[-1] < 4:
            # Create an alpha channel filled with ones (fully opaque)
            mask = torch.ones(image.shape[0], image.shape[1], image.shape[2], dtype=image.dtype, device=image.device)
        else:
            # Extract the specified channel
            mask = image[:, :, :, channel_index]
        
        # Apply clamping if requested
        if clamp_output:
            mask = torch.clamp(mask, 0.0, 1.0)
        
        # Apply normalization if requested
        if normalize_output:
            mask_min = mask.min()
            mask_max = mask.max()
            if mask_max > mask_min:  # Avoid division by zero
                mask = (mask - mask_min) / (mask_max - mask_min)
        
        
        
        return (mask,)


class ImageBlend:
    """
    Blend two images using various blend modes with optional clamping for negative values.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "difference"],),
                "clamp_negative": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend_images"
    CATEGORY = "TiledWan"

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str, clamp_negative: bool):
        """
        Blend two images using the specified blend mode and factor.
        
        Args:
            image1: Base image tensor
            image2: Overlay image tensor  
            blend_factor: Blending strength (0.0 to 1.0)
            blend_mode: Blending mode to use
            clamp_negative: Whether to clamp values below 0 to 0
            
        Returns:
            tuple: Blended image tensor
        """
        # Fix alpha channels and ensure same device
        image1, image2 = node_helpers.image_alpha_fix(image1, image2)
        image2 = image2.to(image1.device)
        
        # Resize image2 to match image1 if shapes differ
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            image2 = image2.permute(0, 2, 3, 1)

        # Apply blend mode
        blended_image = self.blend_mode(image1, image2, blend_mode)
        
        # Mix with original based on blend factor
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        
        # Apply clamping
        if clamp_negative:
            blended_image = torch.clamp(blended_image, 0.0, max=None)

        return (blended_image,)

    def blend_mode(self, img1, img2, mode):
        """
        Apply the specified blend mode to two images.
        
        Args:
            img1: Base image
            img2: Overlay image
            mode: Blend mode string
            
        Returns:
            torch.Tensor: Blended image
        """
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "screen":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "overlay":
            return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == "soft_light":
            return torch.where(img2 <= 0.5, 
                             img1 - (1 - 2 * img2) * img1 * (1 - img1), 
                             img1 + (2 * img2 - 1) * (self.g(img1) - img1))
        elif mode == "difference":
            return torch.abs(img1 - img2)
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        """
        Helper function for soft light blend mode.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Processed tensor
        """
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TiledWanImageToMask": ImageToMask,
    "TiledWanImageBlend": ImageBlend
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledWanImageToMask": "TiledWan Image To Mask",
    "TiledWanImageBlend": "TiledWan Image Blend"
}
