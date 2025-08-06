import torch
import comfy.utils
import comfy.model_management
import node_helpers
import random
import os
import sys
import importlib
import math
import nodes
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, binary_closing, binary_fill_holes


class ImageToMask:
    """
    Convert an image to a mask by extracting a specific channel.
    Supports red, green, blue, and alpha channels with optional clamping and normalization.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to convert to mask. Can be any standard image format."}),
                "channel": (["red", "green", "blue", "alpha"], {"tooltip": "Which color channel to extract as the mask. Alpha channel only available if image has transparency."}),
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Clamp output values to 0-1 range. Recommended to keep enabled for proper mask values."
                }),
                "normalize_output": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled", 
                    "label_off": "disabled",
                    "tooltip": "Normalize the output to use full 0-1 range."
                }),
            },
        }

    CATEGORY = "TiledWan"
    DESCRIPTION = """
    Extracts a specific color channel from an image to create a mask.
    
    Useful for converting colored regions, alpha channels, or channel-specific data into masks
    for use with inpainting, compositing, or other masking operations. The output mask values
    represent the intensity of the selected channel, with proper normalization and clamping.
    """
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ("Grayscale mask derived from the selected image channel. White areas indicate high channel values, black areas indicate low values.",)
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


class ImageStatistics:
    """
    Calculate and display comprehensive statistics for an image.
    Shows min, max, mean, variance, median, and other useful statistics.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to analyze. Statistics will be calculated across all pixels and channels."}),
                "show_per_channel": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Display separate statistics for each color channel (R, G, B) in addition to overall statistics."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("image", "min_value", "max_value", "mean_value", "variance", "median_value")
    OUTPUT_TOOLTIPS = ("Pass-through of the input image", 
                      "Minimum pixel value found in the image", 
                      "Maximum pixel value found in the image", 
                      "Average pixel value across the entire image", 
                      "Variance of pixel values (measure of spread/contrast)", 
                      "Median pixel value (middle value when sorted)")
    FUNCTION = "calculate_statistics"
    OUTPUT_NODE = True  # This allows the node to display output in the console
    CATEGORY = "TiledWan"
    DESCRIPTION = """
    Analyzes image pixel values and provides comprehensive statistical information.
    
    Calculates key metrics including minimum, maximum, mean, variance, and median values.
    Useful for understanding image characteristics, checking normalization, detecting
    issues with image processing, and debugging workflows. Can show per-channel
    statistics for color analysis.
    """

    def calculate_statistics(self, image: torch.Tensor, show_per_channel: bool):
        """
        Calculate comprehensive statistics for the input image.
        
        Args:
            image: Input image tensor [batch, height, width, channels]
            show_per_channel: Whether to show per-channel statistics
            
        Returns:
            tuple: (original_image, min, max, mean, variance, median)
        """
        print("\n=== TiledWan Image Statistics ===")
        
        # Basic image info
        batch_size, height, width, channels = image.shape
        total_pixels = height * width * channels * batch_size
        
        print(f"Image shape: {image.shape} (B×H×W×C)")
        print(f"Total pixels: {total_pixels:,}")
        print(f"Memory usage: {image.numel() * image.element_size() / (1024*1024):.2f} MB")
        print(f"Data type: {image.dtype}")
        print(f"Device: {image.device}")
        
        # Calculate global statistics
        min_val = image.min().item()
        max_val = image.max().item()
        mean_val = image.mean().item()
        variance_val = image.var().item()
        std_val = image.std().item()
        
        # Calculate median (flatten the tensor first)
        flattened = image.flatten()
        # Sort and find median manually to avoid compatibility issues
        sorted_values = torch.sort(flattened).values
        n = len(sorted_values)
        if n % 2 == 0:
            median_val = (sorted_values[n//2 - 1] + sorted_values[n//2]).item() / 2.0
        else:
            median_val = sorted_values[n//2].item()
        
        # Additional useful statistics
        non_zero_count = torch.count_nonzero(image).item()
        zero_count = total_pixels - non_zero_count
        
        print("\n--- Global Statistics ---")
        print(f"Min value: {min_val:.6f}")
        print(f"Max value: {max_val:.6f}")
        print(f"Mean value: {mean_val:.6f}")
        print(f"Median value: {median_val:.6f}")
        print(f"Variance: {variance_val:.6f}")
        print(f"Standard deviation: {std_val:.6f}")
        print(f"Value range: {max_val - min_val:.6f}")
        print(f"Non-zero pixels: {non_zero_count:,} ({non_zero_count/total_pixels:.2%})")
        print(f"Zero pixels: {zero_count:,} ({zero_count/total_pixels:.2%})")
        
        # Per-channel statistics if requested
        if show_per_channel and channels > 1:
            print("\n--- Per-Channel Statistics ---")
            channel_names = ["Red", "Green", "Blue", "Alpha"][:channels]
            
            for c in range(channels):
                channel_data = image[:, :, :, c]
                ch_min = channel_data.min().item()
                ch_max = channel_data.max().item()
                ch_mean = channel_data.mean().item()
                # Calculate median manually for compatibility
                ch_flattened = channel_data.flatten()
                ch_sorted = torch.sort(ch_flattened).values
                ch_n = len(ch_sorted)
                if ch_n % 2 == 0:
                    ch_median = (ch_sorted[ch_n//2 - 1] + ch_sorted[ch_n//2]).item() / 2.0
                else:
                    ch_median = ch_sorted[ch_n//2].item()
                ch_std = channel_data.std().item()
                
                print(f"{channel_names[c]} Channel:")
                print(f"  Min: {ch_min:.6f}, Max: {ch_max:.6f}")
                print(f"  Mean: {ch_mean:.6f}, Median: {ch_median:.6f}")
                print(f"  Std: {ch_std:.6f}, Range: {ch_max - ch_min:.6f}")
        
        # Value distribution analysis
        print("\n--- Value Distribution ---")
        # Create bins for histogram analysis
        bins = torch.linspace(min_val, max_val, 11)  # 10 bins
        hist = torch.histc(flattened, bins=10, min=min_val, max=max_val)
        
        print("Value distribution (10 bins):")
        for i in range(len(hist)):
            bin_start = bins[i].item()
            bin_end = bins[i+1].item() if i < len(bins)-1 else max_val
            count = hist[i].item()
            percentage = count / total_pixels * 100
            print(f"  [{bin_start:.3f} - {bin_end:.3f}]: {count:,.0f} pixels ({percentage:.1f}%)")
        
        # Percentiles
        print("\n--- Percentiles ---")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        # Use the already sorted values from median calculation
        for p in percentiles:
            idx = int((p / 100.0) * (len(sorted_values) - 1))
            value = sorted_values[idx].item()
            print(f"  {p:2d}th percentile: {value:.6f}")
        
        # Potential issues detection
        print("\n--- Analysis ---")
        if min_val < 0:
            print("⚠️  Warning: Image contains negative values!")
        if max_val > 1:
            print("⚠️  Warning: Image contains values > 1.0!")
        if min_val == max_val:
            print("⚠️  Warning: Image has constant values (no variation)!")
        if std_val < 0.01:
            print("⚠️  Notice: Very low standard deviation - image might be nearly uniform")
        if non_zero_count / total_pixels < 0.1:
            print("⚠️  Notice: Most pixels are zero - sparse image")
        
        print("================================\n")
        
        # Return the image and key statistics
        return (image, min_val, max_val, mean_val, variance_val, median_val)

def rescale_i(samples, width, height, algorithm: str):
    samples = samples.movedim(-1, 1)
    algorithm = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
    samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    samples = samples.movedim(1, -1)
    return samples


def rescale_m(samples, width, height, algorithm: str):
    samples = samples.unsqueeze(1)
    algorithm = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
    samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    samples = samples.squeeze(1)
    return samples


def preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height):
    current_width, current_height = image.shape[2], image.shape[1]  # Image size [batch, height, width, channels]
    
    if preresize_mode == "ensure minimum resolution":
        if current_width >= preresize_min_width and current_height >= preresize_min_height:
            return image, mask, optional_context_mask

        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height

        scale_factor = max(scale_factor_min_width, scale_factor_min_height)

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, upscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'bilinear')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
        
        assert target_width >= preresize_min_width and target_height >= preresize_min_height, \
            f"Internal error: After resizing, target size {target_width}x{target_height} is smaller than min size {preresize_min_width}x{preresize_min_height}"

    elif preresize_mode == "ensure minimum and maximum resolution":
        if preresize_min_width <= current_width <= preresize_max_width and preresize_min_height <= current_height <= preresize_max_height:
            return image, mask, optional_context_mask

        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height
        scale_factor_min = max(scale_factor_min_width, scale_factor_min_height)

        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

        if scale_factor_min > 1 and scale_factor_max < 1:
            assert False, "Cannot meet both minimum and maximum resolution requirements with aspect ratio preservation."
        
        if scale_factor_min > 1:  # We're upscaling to meet min resolution
            scale_factor = scale_factor_min
            rescale_algorithm = upscale_algorithm  # Use upscale algorithm for min resolution
        else:  # We're downscaling to meet max resolution
            scale_factor = scale_factor_max
            rescale_algorithm = downscale_algorithm  # Use downscale algorithm for max resolution

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, rescale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest') # Always nearest for efficiency
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest') # Always nearest for efficiency
        
        assert preresize_min_width <= target_width <= preresize_max_width, \
            f"Internal error: Target width {target_width} is outside the range {preresize_min_width} - {preresize_max_width}"
        assert preresize_min_height <= target_height <= preresize_max_height, \
            f"Internal error: Target height {target_height} is outside the range {preresize_min_height} - {preresize_max_height}"

    elif preresize_mode == "ensure maximum resolution":
        if current_width <= preresize_max_width and current_height <= preresize_max_height:
            return image, mask, optional_context_mask

        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

        target_width = int(current_width * scale_factor_max)
        target_height = int(current_height * scale_factor_max)

        image = rescale_i(image, target_width, target_height, downscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest')  # Always nearest for efficiency
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest')  # Always nearest for efficiency

        assert target_width <= preresize_max_width and target_height <= preresize_max_height, \
            f"Internal error: Target size {target_width}x{target_height} is greater than max size {preresize_max_width}x{preresize_max_height}"

    return image, mask, optional_context_mask


def fillholes_iterative_hipass_fill_m(samples):
    thresholds = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    mask_np = samples.squeeze(0).cpu().numpy()

    for threshold in thresholds:
        thresholded_mask = mask_np >= threshold
        closed_mask = binary_closing(thresholded_mask, structure=np.ones((3, 3)), border_value=1)
        filled_mask = binary_fill_holes(closed_mask)
        mask_np = np.maximum(mask_np, np.where(filled_mask != 0, threshold, 0))

    final_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

    return final_mask


def hipassfilter_m(samples, threshold):
    filtered_mask = samples.clone()
    filtered_mask[filtered_mask < threshold] = 0
    return filtered_mask


def expand_m(mask, pixels):
    sigma = pixels / 4
    mask_np = mask.squeeze(0).cpu().numpy()
    kernel_size = math.ceil(sigma * 1.5 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_mask = grey_dilation(mask_np, footprint=kernel)
    dilated_mask = dilated_mask.astype(np.float32)
    dilated_mask = torch.from_numpy(dilated_mask)
    dilated_mask = torch.clamp(dilated_mask, 0.0, 1.0)
    return dilated_mask.unsqueeze(0)


def invert_m(samples):
    inverted_mask = samples.clone()
    inverted_mask = 1.0 - inverted_mask
    return inverted_mask


def blur_m(samples, pixels):
    mask = samples.squeeze(0)
    sigma = pixels / 4 
    mask_np = mask.cpu().numpy()
    blurred_mask = gaussian_filter(mask_np, sigma=sigma)
    blurred_mask = torch.from_numpy(blurred_mask).float()
    blurred_mask = torch.clamp(blurred_mask, 0.0, 1.0)
    return blurred_mask.unsqueeze(0)


def extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
    B, H, W, C = image.shape

    new_H = int(H * (1.0 + extend_up_factor - 1.0 + extend_down_factor - 1.0))
    new_W = int(W * (1.0 + extend_left_factor - 1.0 + extend_right_factor - 1.0))

    assert new_H >= 0, f"Error: Trying to crop too much, height ({new_H}) must be >= 0"
    assert new_W >= 0, f"Error: Trying to crop too much, width ({new_W}) must be >= 0"

    expanded_image = torch.zeros(1, new_H, new_W, C, device=image.device)
    expanded_mask = torch.ones(1, new_H, new_W, device=mask.device)
    expanded_optional_context_mask = torch.zeros(1, new_H, new_W, device=optional_context_mask.device)

    up_padding = int(H * (extend_up_factor - 1.0))
    down_padding = new_H - H - up_padding
    left_padding = int(W * (extend_left_factor - 1.0))
    right_padding = new_W - W - left_padding

    slice_target_up = max(0, up_padding)
    slice_target_down = min(new_H, up_padding + H)
    slice_target_left = max(0, left_padding)
    slice_target_right = min(new_W, left_padding + W)

    slice_source_up = max(0, -up_padding)
    slice_source_down = min(H, new_H - up_padding)
    slice_source_left = max(0, -left_padding)
    slice_source_right = min(W, new_W - left_padding)

    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    expanded_image[:, :, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = image[:, :, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    if up_padding > 0:
        expanded_image[:, :, :up_padding, slice_target_left:slice_target_right] = image[:, :, 0:1, slice_source_left:slice_source_right].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, slice_target_left:slice_target_right] = image[:, :, -1:, slice_source_left:slice_source_right].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, :, :left_padding] = expanded_image[:, :, :, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, :, -right_padding:] = expanded_image[:, :, :, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    expanded_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    expanded_optional_context_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = optional_context_mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]

    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    return expanded_image, expanded_mask, expanded_optional_context_mask


def debug_context_location_in_image(image, x, y, w, h):
    debug_image = image.clone()
    debug_image[:, y:y+h, x:x+w, :] = 1.0 - debug_image[:, y:y+h, x:x+w, :]
    return debug_image


def findcontextarea_m(mask):
    mask_squeezed = mask[0]  # Now shape is [H, W]
    non_zero_indices = torch.nonzero(mask_squeezed)

    H, W = mask_squeezed.shape

    if non_zero_indices.numel() == 0:
        x, y = -1, -1
        w, h = -1, -1
    else:
        y = torch.min(non_zero_indices[:, 0]).item()
        x = torch.min(non_zero_indices[:, 1]).item()
        y_max = torch.max(non_zero_indices[:, 0]).item()
        x_max = torch.max(non_zero_indices[:, 1]).item()
        w = x_max - x + 1  # +1 to include the max index
        h = y_max - y + 1  # +1 to include the max index

    context = mask[:, y:y+h, x:x+w]
    return context, x, y, w, h


def growcontextarea_m(context, mask, x, y, w, h, extend_factor):
    img_h, img_w = mask.shape[1], mask.shape[2]

    # Compute intended growth in each direction
    grow_left = int(round(w * (extend_factor-1.0) / 2.0))
    grow_right = int(round(w * (extend_factor-1.0) / 2.0))
    grow_up = int(round(h * (extend_factor-1.0) / 2.0))
    grow_down = int(round(h * (extend_factor-1.0) / 2.0))

    # Try to grow left, but clamp at 0
    new_x = x - grow_left
    if new_x < 0:
        new_x = 0

    # Try to grow up, but clamp at 0
    new_y = y - grow_up
    if new_y < 0:
        new_y = 0

    # Right edge
    new_x2 = x + w + grow_right
    if new_x2 > img_w:
        new_x2 = img_w

    # Bottom edge
    new_y2 = y + h + grow_down
    if new_y2 > img_h:
        new_y2 = img_h

    # New width and height
    new_w = new_x2 - new_x
    new_h = new_y2 - new_y

    # Extract the context
    new_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]

    if new_h < 0 or new_w < 0:
        new_x = 0
        new_y = 0
        new_w = mask.shape[2]
        new_h = mask.shape[1]

    return new_context, new_x, new_y, new_w, new_h


def combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask):
    _, x_opt, y_opt, w_opt, h_opt = findcontextarea_m(optional_context_mask)
    if x == -1:
        x, y, w, h = x_opt, y_opt, w_opt, h_opt
    if x_opt == -1:
        x_opt, y_opt, w_opt, h_opt = x, y, w, h
    if x == -1:
        return torch.zeros(1, 0, 0, device=mask.device), -1, -1, -1, -1
    new_x = min(x, x_opt)
    new_y = min(y, y_opt)
    new_x_max = max(x + w, x_opt + w_opt)
    new_y_max = max(y + h, y_opt + h_opt)
    new_w = new_x_max - new_x
    new_h = new_y_max - new_y
    combined_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]
    return combined_context, new_x, new_y, new_w, new_h


def pad_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def crop_magic_im(image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm):
    image = image.clone()
    mask = mask.clone()
    
    # Ok this is the most complex function in this node. The one that does the magic after all the preparation done by the other nodes.
    # Basically this function determines the right context area that encompasses the whole context area (mask+optional_context_mask),
    # that is ideally within the bounds of the original image, and that has the right aspect ratio to match target width and height.
    # It may grow the image if the aspect ratio wouldn't fit in the original image.
    # It keeps track of that growing to then be able to crop the image in the stitch node.
    # Finally, it crops the context area and resizes it to be exactly target_w and target_h.
    # It keeps track of that resize to be able to revert it in the stitch node.

    # Check for invalid inputs
    if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
        return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

    # Step 1: Pad target dimensions to be multiples of padding
    if padding != 0:
        target_w = pad_to_multiple(target_w, padding)
        target_h = pad_to_multiple(target_h, padding)

    # Step 2: Calculate target aspect ratio
    target_aspect_ratio = target_w / target_h

    # Step 3: Grow current context area to meet the target aspect ratio
    B, image_h, image_w, C = image.shape
    context_aspect_ratio = w / h
    if context_aspect_ratio < target_aspect_ratio:
        # Grow width to meet aspect ratio
        new_w = int(h * target_aspect_ratio)
        new_h = h
        new_x = x - (new_w - w) // 2
        new_y = y

        # Adjust new_x to keep within bounds
        if new_x < 0:
            shift = -new_x
            if new_x + new_w + shift <= image_w:
                new_x += shift
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow
        elif new_x + new_w > image_w:
            overflow = new_x + new_w - image_w
            if new_x - overflow >= 0:
                new_x -= overflow
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow

    else:
        # Grow height to meet aspect ratio
        new_w = w
        new_h = int(w / target_aspect_ratio)
        new_x = x
        new_y = y - (new_h - h) // 2

        # Adjust new_y to keep within bounds
        if new_y < 0:
            shift = -new_y
            if new_y + new_h + shift <= image_h:
                new_y += shift
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow
        elif new_y + new_h > image_h:
            overflow = new_y + new_h - image_h
            if new_y - overflow >= 0:
                new_y -= overflow
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow

    # Step 4: Grow the image to accommodate the new context area
    up_padding, down_padding, left_padding, right_padding = 0, 0, 0, 0

    expanded_image_w = image_w
    expanded_image_h = image_h

    # Adjust width for left overflow (x < 0) and right overflow (x + w > image_w)
    if new_x < 0:
        left_padding = -new_x
        expanded_image_w += left_padding
    if new_x + new_w > image_w:
        right_padding = (new_x + new_w - image_w)
        expanded_image_w += right_padding
    # Adjust height for top overflow (y < 0) and bottom overflow (y + h > image_h)
    if new_y < 0:
        up_padding = -new_y
        expanded_image_h += up_padding 
    if new_y + new_h > image_h:
        down_padding = (new_y + new_h - image_h)
        expanded_image_h += down_padding

    # Step 5: Create the new image and mask
    expanded_image = torch.zeros((image.shape[0], expanded_image_h, expanded_image_w, image.shape[3]), device=image.device)
    expanded_mask = torch.ones((mask.shape[0], expanded_image_h, expanded_image_w), device=mask.device)

    # Reorder the tensors to match the required dimension format for padding
    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    # Ensure the expanded image has enough room to hold the padded version of the original image
    expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = image

    # Fill the new extended areas with the edge values of the image
    if up_padding > 0:
        expanded_image[:, :, :up_padding, left_padding:left_padding + image_w] = image[:, :, 0:1, left_padding:left_padding + image_w].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, left_padding:left_padding + image_w] = image[:, :, -1:, left_padding:left_padding + image_w].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, :left_padding] = expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, -right_padding:] = expanded_image[:, :, up_padding:up_padding + image_h, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    # Reorder the tensors back to [B, H, W, C] format
    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    # Same for the mask
    expanded_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

    # Record the cto values (canvas to original)
    cto_x = left_padding
    cto_y = up_padding
    cto_w = image_w
    cto_h = image_h

    # The final expanded image and mask
    canvas_image = expanded_image
    canvas_mask = expanded_mask

    # Step 6: Crop the image and mask around x, y, w, h
    ctc_x = new_x+left_padding
    ctc_y = new_y+up_padding
    ctc_w = new_w
    ctc_h = new_h

    # Crop the image and mask
    cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    # Step 7: Resize image and mask to the target width and height
    # Decide which algorithm to use based on the scaling direction
    if target_w > ctc_w or target_h > ctc_h:  # Upscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, upscale_algorithm)
    else:  # Downscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, downscale_algorithm)

    return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h


def stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
    canvas_image = canvas_image.clone()
    inpainted_image = inpainted_image.clone()
    mask = mask.clone()

    # Resize inpainted image and mask to match the context size
    _, h, w, _ = inpainted_image.shape
    if ctc_w > w or ctc_h > h:  # Upscaling
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
    else:  # Downscaling
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)

    # Clamp mask to [0, 1] and expand to match image channels
    resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)  # shape: [1, H, W, 1]

    # Extract the canvas region we're about to overwrite
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    # Blend: new = mask * inpainted + (1 - mask) * canvas
    blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop

    # Paste the blended region back onto the canvas
    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended

    # Final crop to get back the original image area
    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]

    return output_image


class MaskStatistics:
    """
    Calculate and display comprehensive statistics for a mask.
    Shows min, max, mean, variance, median, and mask-specific statistics.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Input mask to analyze. Should be a grayscale mask with values typically between 0-1."}),
                "analyze_connected_components": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Analyze connected regions in the mask to count separate masked areas and find the largest connected component."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("mask", "min_value", "max_value", "mean_value", "variance", "median_value", "white_pixel_count")
    OUTPUT_TOOLTIPS = ("Pass-through of the input mask", 
                      "Minimum mask value found", 
                      "Maximum mask value found", 
                      "Average mask value (coverage ratio)", 
                      "Variance of mask values", 
                      "Median mask value", 
                      "Number of white/masked pixels (where value > 0.5)")
    FUNCTION = "calculate_mask_statistics"
    OUTPUT_NODE = True  # This allows the node to display output in the console
    CATEGORY = "TiledWan"
    DESCRIPTION = """
    Analyzes mask properties and provides detailed statistical information.
    
    Calculates standard statistics plus mask-specific metrics like coverage ratio,
    connected component analysis, and white pixel count. Essential for understanding
    mask quality, coverage area, and structure. Helps debug masking operations and
    validate mask inputs for inpainting or compositing workflows.
    """

    def calculate_mask_statistics(self, mask: torch.Tensor, analyze_connected_components: bool):
        """
        Calculate comprehensive statistics for the input mask.
        
        Args:
            mask: Input mask tensor [batch, height, width]
            analyze_connected_components: Whether to analyze connected components
            
        Returns:
            tuple: (original_mask, min, max, mean, variance, median, white_pixel_count)
        """
        print("\n=== TiledWan Mask Statistics ===")
        
        # Basic mask info
        if len(mask.shape) == 3:
            batch_size, height, width = mask.shape
            channels = 1
        else:
            batch_size, height, width, channels = mask.shape
            
        total_pixels = height * width * batch_size
        
        print(f"Mask shape: {mask.shape} (B×H×W)")
        print(f"Total pixels: {total_pixels:,}")
        print(f"Memory usage: {mask.numel() * mask.element_size() / (1024*1024):.2f} MB")
        print(f"Data type: {mask.dtype}")
        print(f"Device: {mask.device}")
        
        # Calculate global statistics
        min_val = mask.min().item()
        max_val = mask.max().item()
        mean_val = mask.mean().item()
        variance_val = mask.var().item()
        std_val = mask.std().item()
        
        # Calculate median (flatten the tensor first)
        flattened = mask.flatten()
        # Sort and find median manually to avoid compatibility issues
        sorted_values = torch.sort(flattened).values
        n = len(sorted_values)
        if n % 2 == 0:
            median_val = (sorted_values[n//2 - 1] + sorted_values[n//2]).item() / 2.0
        else:
            median_val = sorted_values[n//2].item()
        
        # Mask-specific statistics
        white_pixels = torch.sum(mask > 0.5).item()  # Pixels considered "white" (active)
        black_pixels = total_pixels - white_pixels
        coverage = white_pixels / total_pixels
        
        # Count unique values (useful for masks)
        unique_values = torch.unique(mask)
        unique_count = len(unique_values)
        
        print("\n--- Global Statistics ---")
        print(f"Min value: {min_val:.6f}")
        print(f"Max value: {max_val:.6f}")
        print(f"Mean value: {mean_val:.6f}")
        print(f"Median value: {median_val:.6f}")
        print(f"Variance: {variance_val:.6f}")
        print(f"Standard deviation: {std_val:.6f}")
        print(f"Value range: {max_val - min_val:.6f}")
        
        print("\n--- Mask-Specific Statistics ---")
        print(f"White pixels (>0.5): {white_pixels:,} ({coverage:.2%})")
        print(f"Black pixels (≤0.5): {black_pixels:,} ({(1-coverage):.2%})")
        print(f"Unique values: {unique_count}")
        if unique_count <= 10:
            print(f"Unique values list: {[f'{val:.3f}' for val in unique_values.tolist()]}")
        
        # Mask density analysis
        print(f"Mask density: {coverage:.4f}")
        if coverage > 0.8:
            print("  → High density mask (mostly white)")
        elif coverage > 0.5:
            print("  → Medium-high density mask")
        elif coverage > 0.2:
            print("  → Medium-low density mask")
        elif coverage > 0.05:
            print("  → Low density mask")
        else:
            print("  → Very sparse mask")
        
        # Value distribution analysis
        print("\n--- Value Distribution ---")
        # For masks, show distribution more relevant to binary/grayscale nature
        if unique_count <= 20:  # If few unique values, show exact counts
            print("Exact value counts:")
            for val in unique_values:
                count = torch.sum(mask == val).item()
                percentage = count / total_pixels * 100
                print(f"  Value {val:.3f}: {count:,} pixels ({percentage:.1f}%)")
        else:
            # Create bins for histogram analysis
            bins = torch.linspace(min_val, max_val, 11)  # 10 bins
            hist = torch.histc(flattened, bins=10, min=min_val, max=max_val)
            
            print("Value distribution (10 bins):")
            for i in range(len(hist)):
                bin_start = bins[i].item()
                bin_end = bins[i+1].item() if i < len(bins)-1 else max_val
                count = hist[i].item()
                percentage = count / total_pixels * 100
                print(f"  [{bin_start:.3f} - {bin_end:.3f}]: {count:,.0f} pixels ({percentage:.1f}%)")
        
        # Percentiles
        print("\n--- Percentiles ---")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        # Use the already sorted values from median calculation
        for p in percentiles:
            idx = int((p / 100.0) * (len(sorted_values) - 1))
            value = sorted_values[idx].item()
            print(f"  {p:2d}th percentile: {value:.6f}")
        
        # Connected components analysis (if requested)
        if analyze_connected_components and coverage > 0 and coverage < 1:
            print("\n--- Connected Components Analysis ---")
            try:
                # Simple connected components analysis
                # For each batch item
                total_components = 0
                largest_component = 0
                smallest_component = float('inf')
                
                for b in range(batch_size):
                    mask_slice = mask[b] if len(mask.shape) == 3 else mask[b, :, :, 0]
                    binary_mask = (mask_slice > 0.5).float()
                    
                    if binary_mask.sum() > 0:  # Only analyze if there are white pixels
                        # Simple component counting (not perfect but gives an idea)
                        # Count clusters by looking at transitions
                        transitions = 0
                        prev_row_white = torch.zeros(width, dtype=torch.bool, device=mask.device)
                        
                        for row in range(height):
                            current_row = binary_mask[row] > 0.5
                            # Count horizontal transitions in this row
                            if row == 0:
                                transitions += torch.sum(current_row[1:] != current_row[:-1]).item()
                            
                            # Count vertical transitions
                            transitions += torch.sum(current_row != prev_row_white).item()
                            prev_row_white = current_row
                        
                        # Rough estimate of components (very approximate)
                        estimated_components = max(1, transitions // 4)
                        total_components += estimated_components
                        
                        component_size = binary_mask.sum().item()
                        largest_component = max(largest_component, component_size)
                        smallest_component = min(smallest_component, component_size)
                
                if total_components > 0:
                    avg_components = total_components / batch_size
                    print(f"Estimated components per image: {avg_components:.1f}")
                    print(f"Largest component: {largest_component:,} pixels")
                    if smallest_component != float('inf'):
                        print(f"Smallest component: {smallest_component:,} pixels")
                else:
                    print("No connected components found")
                    
            except Exception as e:
                print(f"Connected components analysis failed: {str(e)}")
        
        # Potential issues detection
        print("\n--- Analysis ---")
        if min_val < 0:
            print("⚠️  Warning: Mask contains negative values!")
        if max_val > 1:
            print("⚠️  Warning: Mask contains values > 1.0!")
        if min_val == max_val:
            print("⚠️  Warning: Mask has constant values (uniform mask)!")
        if unique_count == 2 and min_val == 0 and max_val == 1:
            print("✓  Perfect binary mask (0 and 1 only)")
        elif unique_count <= 5:
            print(f"✓  Nearly binary mask ({unique_count} unique values)")
        if coverage < 0.01:
            print("⚠️  Notice: Very sparse mask - almost empty")
        if coverage > 0.99:
            print("⚠️  Notice: Very dense mask - almost full")
        if std_val < 0.01:
            print("⚠️  Notice: Very low standard deviation - mask might be nearly uniform")
        
        print("================================\n")
        
        # Return the mask and key statistics
        return (mask, min_val, max_val, mean_val, variance_val, median_val, white_pixels)


class InpaintCropImproved:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Required inputs
                "image": ("IMAGE", {"tooltip": "Input image(s) to be processed. Can be a single image or batch of images for video processing."}),

                # Resize algorithms
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bilinear", "tooltip": "Algorithm used when downscaling images. Bilinear provides good balance of quality and speed."}),
                "upscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bicubic", "tooltip": "Algorithm used when upscaling images. Bicubic provides better quality for upscaling."}),

                # Pre-resize input image
                "preresize": ("BOOLEAN", {"default": False, "tooltip": "Enable to resize the original image before processing. Useful for normalizing input sizes."}),
                "preresize_mode": (["ensure minimum resolution", "ensure maximum resolution", "ensure minimum and maximum resolution"], {"default": "ensure minimum resolution", "tooltip": "Mode for pre-resizing: minimum ensures image is at least the specified size, maximum caps the size, both enforces a range."}),
                "preresize_min_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Minimum width for pre-resize operation. Image will be upscaled if smaller."}),
                "preresize_min_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Minimum height for pre-resize operation. Image will be upscaled if smaller."}),
                "preresize_max_width": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Maximum width for pre-resize operation. Image will be downscaled if larger."}),
                "preresize_max_height": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Maximum height for pre-resize operation. Image will be downscaled if larger."}),

                # Mask manipulation
                "mask_fill_holes": ("BOOLEAN", {"default": True, "tooltip": "Fill holes in the mask using iterative morphological operations. Helps create more complete masked regions."}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Expand the mask by this many pixels before processing. Useful for including more context around masked areas."}),
                "mask_invert": ("BOOLEAN", {"default": False, "tooltip": "Invert the mask so that masked areas become unmasked and vice versa."}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1, "tooltip": "Create a soft transition zone around mask edges. Higher values create smoother blending during stitching."}),
                "mask_hipass_filter": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01, "tooltip": "Remove mask values below this threshold. Helps eliminate weak mask areas and noise."}),

                # Extend image for outpainting
                "extend_for_outpainting": ("BOOLEAN", {"default": False, "tooltip": "Extend the image canvas for outpainting. Adds padding around the image for generating content beyond original boundaries."}),
                "extend_up_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Factor to extend image upward. 1.0 = no extension, 1.5 = 50% extension upward."}),
                "extend_down_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Factor to extend image downward. 1.0 = no extension, 1.5 = 50% extension downward."}),
                "extend_left_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Factor to extend image leftward. 1.0 = no extension, 1.5 = 50% extension leftward."}),
                "extend_right_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Factor to extend image rightward. 1.0 = no extension, 1.5 = 50% extension rightward."}),

                # Context
                "context_from_mask_extend_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01, "tooltip": "Expand the context area around the mask by this factor. 1.2 = 20% expansion in all directions. Larger values include more surrounding context for better inpainting."}),

                # Output
                "output_resize_to_target_size": ("BOOLEAN", {"default": True, "tooltip": "Resize the output to specific dimensions. When disabled, output size depends on mask area and extend factor."}),
                "output_target_width": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Target width for output image when resize to target size is enabled. Should match your inpainting model's preferred resolution."}),
                "output_target_height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Target height for output image when resize to target size is enabled. Should match your inpainting model's preferred resolution."}),
                "output_padding": (["0", "8", "16", "32", "64", "128", "256", "512"], {"default": "32", "tooltip": "Padding to ensure output dimensions are multiples of this value. Important for models that require specific dimension alignment (e.g., 8 for VAE, 32 for some diffusion models)."}),
                
                # Batch consistency
                "keep_window_size": ("BOOLEAN", {"default": False, "tooltip": "Maintain consistent crop window size across all images in batch. Essential for video processing to avoid flickering. Uses maximum dimensions found and interpolates missing coordinates."}),
           },
           "optional": {
                # Optional inputs
                "mask": ("MASK", {"tooltip": "Mask defining areas to be inpainted. White areas will be inpainted, black areas will be preserved. If not provided, the entire image will be processed."}),
                "optional_context_mask": ("MASK", {"tooltip": "Additional mask defining extra context areas to include in the crop. Useful for ensuring important surrounding details are preserved during inpainting."}),
           }
        }

    FUNCTION = "inpaint_crop"
    CATEGORY = "inpaint"
    DESCRIPTION = """
    Advanced inpainting crop node that intelligently crops images around masked areas for optimal inpainting results.
    
    Features:
    - Batch processing with temporal consistency for video sequences
    - Automatic context area detection and expansion
    - Smart resizing with aspect ratio preservation
    - Optional pre-processing (resize, mask manipulation, outpainting extension)
    - Window size consistency across batches for video processing
    - Linear interpolation for smooth mask transitions in video sequences
    
    The node processes masks to find the optimal crop area, applies various preprocessing steps, 
    and outputs cropped images ready for inpainting along with stitcher data for reconstruction.
    """


    # Remove the following # to turn on debug mode (extra outputs, print statements)
    #'''
    DEBUG_MODE = False
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask")
    OUTPUT_TOOLTIPS = ("Stitcher data containing all information needed to reconstruct the original image after inpainting. Pass this to InpaintStitchImproved.", 
                      "Cropped and processed image ready for inpainting. Contains the masked area plus surrounding context.", 
                      "Processed mask corresponding to the cropped image. Shows which areas need to be inpainted.")

    '''
    
    DEBUG_MODE = True # TODO
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK",
        # DEBUG
        "IMAGE",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "IMAGE",
        "MASK",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask",
        # DEBUG
        "DEBUG_preresize_image",
        "DEBUG_preresize_mask",
        "DEBUG_fillholes_mask",
        "DEBUG_expand_mask",
        "DEBUG_invert_mask",
        "DEBUG_blur_mask",
        "DEBUG_hipassfilter_mask",
        "DEBUG_extend_image",
        "DEBUG_extend_mask",
        "DEBUG_context_from_mask",
        "DEBUG_context_from_mask_location",
        "DEBUG_context_expand",
        "DEBUG_context_expand_location",
        "DEBUG_context_with_context_mask",
        "DEBUG_context_with_context_mask_location",
        "DEBUG_context_to_target",
        "DEBUG_context_to_target_location",
        "DEBUG_context_to_target_image",
        "DEBUG_context_to_target_mask",
        "DEBUG_canvas_image",
        "DEBUG_orig_in_canvas_location",
        "DEBUG_cropped_in_canvas_location",
        "DEBUG_cropped_mask_blend",
    )
    #'''

 
    def inpaint_crop(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, keep_window_size, mask=None, optional_context_mask=None):
        image = image.clone()
        if mask is not None:
            mask = mask.clone()
        if optional_context_mask is not None:
            optional_context_mask = optional_context_mask.clone()

        output_padding = int(output_padding)
        
        # Check that some parameters make sense
        if preresize and preresize_mode == "ensure minimum and maximum resolution":
            assert preresize_max_width >= preresize_min_width, "Preresize maximum width must be greater than or equal to minimum width"
            assert preresize_max_height >= preresize_min_height, "Preresize maximum height must be greater than or equal to minimum height"

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch input')
            print(image.shape, type(image), image.dtype)
            if mask is not None:
                print(mask.shape, type(mask), mask.dtype)
            if optional_context_mask is not None:
                print(optional_context_mask.shape, type(optional_context_mask), optional_context_mask.dtype)

        if image.shape[0] > 1 and not keep_window_size:
            assert output_resize_to_target_size, "output_resize_to_target_size must be enabled when input is a batch of images without keep_window_size enabled, given all images in the batch output have to be the same size (unless keep_window_size is enabled)"

        # When a LoadImage node passes a mask without user editing, it may be the wrong shape.
        # Detect and fix that to avoid shape mismatch errors.
        if mask is not None and (image.shape[0] == 1 or mask.shape[0] == 1 or mask.shape[0] == image.shape[0]):
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(mask) == 0:
                    mask = torch.zeros((mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        if optional_context_mask is not None and (image.shape[0] == 1 or optional_context_mask.shape[0] == 1 or optional_context_mask.shape[0] == image.shape[0]):
            if optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(optional_context_mask) == 0:
                    optional_context_mask = torch.zeros((optional_context_mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        # If no mask is provided, create one with the shape of the image
        if mask is None:
            mask = torch.zeros_like(image[:, :, :, 0])
    
        # If there is only one image for many masks, replicate it for all masks
        if mask.shape[0] > 1 and image.shape[0] == 1:
            assert image.dim() == 4, f"Expected 4D BHWC image tensor, got {image.shape}"
            image = image.expand(mask.shape[0], -1, -1, -1).clone()

        # If there is only one mask for many images, replicate it for all images
        if image.shape[0] > 1 and mask.shape[0] == 1:
            assert mask.dim() == 3, f"Expected 3D BHW mask tensor, got {mask.shape}"
            mask = mask.expand(image.shape[0], -1, -1).clone()

        # If no optional_context_mask is provided, create one with the shape of the image
        if optional_context_mask is None:
            optional_context_mask = torch.zeros_like(image[:, :, :, 0])

        # If there is only one optional_context_mask for many images, replicate it for all images
        if image.shape[0] > 1 and optional_context_mask.shape[0] == 1:
            assert optional_context_mask.dim() == 3, f"Expected 3D BHW optional_context_mask tensor, got {optional_context_mask.shape}"
            optional_context_mask = optional_context_mask.expand(image.shape[0], -1, -1).clone()

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch ready')
            print(image.shape, type(image), image.dtype)
            print(mask.shape, type(mask), mask.dtype)
            print(optional_context_mask.shape, type(optional_context_mask), optional_context_mask.dtype)

         # Validate data
        assert image.ndimension() == 4, f"Expected 4 dimensions for image, got {image.ndimension()}"
        assert mask.ndimension() == 3, f"Expected 3 dimensions for mask, got {mask.ndimension()}"
        assert optional_context_mask.ndimension() == 3, f"Expected 3 dimensions for optional_context_mask, got {optional_context_mask.ndimension()}"
        assert mask.shape[1:] == image.shape[1:3], f"Mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {mask.shape[1:]}"
        assert optional_context_mask.shape[1:] == image.shape[1:3], f"optional_context_mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {optional_context_mask.shape[1:]}"
        assert mask.shape[0] == image.shape[0], f"Mask batch does not match image batch. Expected {image.shape[0]}, got {mask.shape[0]}"
        assert optional_context_mask.shape[0] == image.shape[0], f"Optional context mask batch does not match image batch. Expected {image.shape[0]}, got {optional_context_mask.shape[0]}"

        # Run for each image separately
        result_stitcher = {
            'downscale_algorithm': downscale_algorithm,
            'upscale_algorithm': upscale_algorithm,
            'blend_pixels': mask_blend_pixels,
            'canvas_to_orig_x': [],
            'canvas_to_orig_y': [],
            'canvas_to_orig_w': [],
            'canvas_to_orig_h': [],
            'canvas_image': [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
        }
        
        result_image = []
        result_mask = []

        debug_outputs = {name: [] for name in self.RETURN_NAMES if name.startswith("DEBUG_")}

        batch_size = image.shape[0]
        
        # Compute all contexts first
        batch_contexts, max_w, max_h = self.compute_batch_contexts(
            image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm,
            preresize, preresize_mode, preresize_min_width, preresize_min_height, 
            preresize_max_width, preresize_max_height, extend_for_outpainting,
            extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor,
            mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert,
            mask_blend_pixels, context_from_mask_extend_factor, keep_window_size
        )
        
        # Process each image with pre-computed contexts
        for b in range(batch_size):
            ctx = batch_contexts[b]
            
            outputs = self.inpaint_crop_single_image_with_context(
                ctx['processed_image'], ctx['processed_mask'], ctx['processed_optional_context_mask'],
                ctx['x'], ctx['y'], ctx['w'], ctx['h'], ctx['context'],
                downscale_algorithm, upscale_algorithm, mask_blend_pixels,
                output_resize_to_target_size, output_target_width, output_target_height, output_padding,
                keep_window_size, max_w, max_h, context_from_mask_extend_factor)

            stitcher, cropped_image, cropped_mask = outputs[:3]
            for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                result_stitcher[key].append(stitcher[key])

            cropped_image = cropped_image.clone().squeeze(0)
            result_image.append(cropped_image)
            cropped_mask = cropped_mask.clone().squeeze(0)
            result_mask.append(cropped_mask)

            # Handle the DEBUG_ fields dynamically
            for name, output in zip(self.RETURN_NAMES[3:], outputs[3:]):  # Start from index 3 since first 3 are fixed
                if name.startswith("DEBUG_"):
                    output_array = output.squeeze(0)  # Assuming output needs to be squeezed similar to image/mask
                    debug_outputs[name].append(output_array)

        result_image = torch.stack(result_image, dim=0)
        result_mask = torch.stack(result_mask, dim=0)

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch output')
            print(result_image.shape, type(result_image), result_image.dtype)
            print(result_mask.shape, type(result_mask), result_mask.dtype)

        debug_outputs = {name: torch.stack(values, dim=0) for name, values in debug_outputs.items()}

        return result_stitcher, result_image, result_mask, [debug_outputs[name] for name in self.RETURN_NAMES if name.startswith("DEBUG_")]

    def compute_batch_contexts(self, image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, 
                             preresize, preresize_mode, preresize_min_width, preresize_min_height, 
                             preresize_max_width, preresize_max_height, extend_for_outpainting, 
                             extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor,
                             mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, 
                             mask_blend_pixels, context_from_mask_extend_factor, keep_window_size):
        """
        Compute contexts for all images in the batch before processing.
        Returns processed masks, contexts, and coordinates for each image.
        """
        batch_size = image.shape[0]
        batch_contexts = []
        
        # Step 1: Apply mask preprocessing for all images
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)
            one_mask = mask[b].unsqueeze(0)
            one_optional_context_mask = optional_context_mask[b].unsqueeze(0)
            
            # Apply all mask preprocessing steps
            if preresize:
                one_image, one_mask, one_optional_context_mask = preresize_imm(
                    one_image, one_mask, one_optional_context_mask, downscale_algorithm, 
                    upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, 
                    preresize_max_width, preresize_max_height)
            
            if mask_fill_holes:
                one_mask = fillholes_iterative_hipass_fill_m(one_mask)
            
            if mask_expand_pixels > 0:
                one_mask = expand_m(one_mask, mask_expand_pixels)
            
            if mask_invert:
                one_mask = invert_m(one_mask)
            
            if mask_blend_pixels > 0:
                one_mask = expand_m(one_mask, mask_blend_pixels)
                one_mask = blur_m(one_mask, mask_blend_pixels*0.5)
            
            if mask_hipass_filter >= 0.01:
                one_mask = hipassfilter_m(one_mask, mask_hipass_filter)
                one_optional_context_mask = hipassfilter_m(one_optional_context_mask, mask_hipass_filter)
            
            if extend_for_outpainting:
                one_image, one_mask, one_optional_context_mask = extend_imm(
                    one_image, one_mask, one_optional_context_mask, extend_up_factor, 
                    extend_down_factor, extend_left_factor, extend_right_factor)
            
            batch_contexts.append({
                'processed_image': one_image,
                'processed_mask': one_mask, 
                'processed_optional_context_mask': one_optional_context_mask
            })
        
        # Step 2: Find initial contexts without applying context operations yet
        initial_contexts = []
        for ctx in batch_contexts:
            context, x, y, w, h = findcontextarea_m(ctx['processed_mask'])
            initial_contexts.append({'x': x, 'y': y, 'w': w, 'h': h})
        
        # Step 3: Apply keep_window_size logic if enabled
        max_w = None
        max_h = None
        if keep_window_size and batch_size > 1:
            # Find maximum dimensions from initial contexts
            max_w = max(ctx['w'] for ctx in initial_contexts)
            max_h = max(ctx['h'] for ctx in initial_contexts)
            
  
            # Update all initial contexts to use consistent window size
            # First, interpolate x and y coordinates for frames with invalid values (-1)
            def interpolate_coordinates(coords):
                """Interpolate missing coordinates (-1) using linear interpolation between valid values"""
                coords = coords.copy()
                n = len(coords)
                
                # Find all valid indices (where coord != -1)
                valid_indices = [i for i, coord in enumerate(coords) if coord != -1]
                
                if not valid_indices:
                    # No valid coordinates, fill with zeros
                    return [0] * n
                
                # Handle leading -1s: use first valid value
                first_valid_idx = valid_indices[0]
                first_valid_value = coords[first_valid_idx]
                for i in range(first_valid_idx):
                    coords[i] = first_valid_value
                
                # Handle trailing -1s: use last valid value
                last_valid_idx = valid_indices[-1]
                last_valid_value = coords[last_valid_idx]
                for i in range(last_valid_idx + 1, n):
                    coords[i] = last_valid_value
                
                # Interpolate between valid values
                for i in range(len(valid_indices) - 1):
                    start_idx = valid_indices[i]
                    end_idx = valid_indices[i + 1]
                    start_val = coords[start_idx]
                    end_val = coords[end_idx]
                    
                    # Linear interpolation for indices between start_idx and end_idx
                    for j in range(start_idx + 1, end_idx):
                        alpha = (j - start_idx) / (end_idx - start_idx)
                        coords[j] = int(start_val + alpha * (end_val - start_val))
                
                return coords
            
            # Extract x and y coordinates
            x_coords = [ctx['x'] for ctx in initial_contexts]
            y_coords = [ctx['y'] for ctx in initial_contexts]
            
            # Interpolate invalid coordinates
            x_coords = interpolate_coordinates(x_coords)
            y_coords = interpolate_coordinates(y_coords)
            
            # Update contexts with interpolated coordinates
            for i, ctx in enumerate(initial_contexts):
                ctx['x'] = x_coords[i]
                ctx['y'] = y_coords[i]
                
                # Update dimensions to maximum
                ctx['w'] = max_w
                ctx['h'] = max_h
                
                # Ensure we don't go outside image boundaries
                img_h, img_w = batch_contexts[i]['processed_image'].shape[1], batch_contexts[i]['processed_image'].shape[2]
                if ctx['x'] + ctx['w'] > img_w:
                    ctx['x'] = max(0, img_w - ctx['w'])
                if ctx['y'] + ctx['h'] > img_h:
                    ctx['y'] = max(0, img_h - ctx['h'])
        
        # Step 4: Now apply context operations (grow context, combine with optional mask) after window size keeping
        for i, ctx in enumerate(batch_contexts):
            x, y, w, h = initial_contexts[i]['x'], initial_contexts[i]['y'], initial_contexts[i]['w'], initial_contexts[i]['h']
            
            # Find initial context
            context, x, y, w, h = findcontextarea_m(ctx['processed_mask'])
            if x == -1 or w == -1 or h == -1 or y == -1:
                x, y, w, h = initial_contexts[i]['x'], initial_contexts[i]['y'], initial_contexts[i]['w'], initial_contexts[i]['h']
                context = ctx['processed_mask'][:, y:y+h, x:x+w]
            else:
                # Use the coordinates from keep_window_size if enabled
                if keep_window_size and batch_size > 1:
                    x, y, w, h = initial_contexts[i]['x'], initial_contexts[i]['y'], initial_contexts[i]['w'], initial_contexts[i]['h']
                    context = ctx['processed_mask'][:, y:y+h, x:x+w]
            
            # Grow context if needed
            if context_from_mask_extend_factor >= 1.01:
                context, x, y, w, h = growcontextarea_m(context, ctx['processed_mask'], x, y, w, h, context_from_mask_extend_factor)
            if x == -1 or w == -1 or h == -1 or y == -1:
                x, y, w, h = 0, 0, ctx['processed_image'].shape[2], ctx['processed_image'].shape[1]
                context = ctx['processed_mask'][:, y:y+h, x:x+w]
            
            # Combine with optional context mask
            context, x, y, w, h = combinecontextmask_m(context, ctx['processed_mask'], x, y, w, h, ctx['processed_optional_context_mask'])
            if x == -1 or w == -1 or h == -1 or y == -1:
                x, y, w, h = 0, 0, ctx['processed_image'].shape[2], ctx['processed_image'].shape[1]
                context = ctx['processed_mask'][:, y:y+h, x:x+w]
            
            # Update the context with final values
            ctx['context'] = context
            ctx['x'] = x
            ctx['y'] = y
            ctx['w'] = w
            ctx['h'] = h
        
        return batch_contexts, max_w, max_h

    def inpaint_crop_single_image_with_context(self, image, mask, optional_context_mask, x, y, w, h, context,
                                             downscale_algorithm, upscale_algorithm, mask_blend_pixels,
                                             output_resize_to_target_size, output_target_width, output_target_height, output_padding, 
                                             keep_window_size=False, max_w=None, max_h=None, context_from_mask_extend_factor=1.0):
        """
        Process a single image with pre-computed context coordinates.
        This is the simplified version that skips all the mask preprocessing since it's already done.
        """
        if self.DEBUG_MODE:
            DEBUG_preresize_image = image.clone()
            DEBUG_preresize_mask = mask.clone()
            DEBUG_fillholes_mask = mask.clone()
            DEBUG_expand_mask = mask.clone()
            DEBUG_invert_mask = mask.clone()
            DEBUG_blur_mask = mask.clone()
            DEBUG_hipassfilter_mask = mask.clone()
            DEBUG_extend_image = image.clone()
            DEBUG_extend_mask = mask.clone()
            DEBUG_context_from_mask = context.clone()
            DEBUG_context_from_mask_location = debug_context_location_in_image(image, x, y, w, h)
            DEBUG_context_expand = context.clone()
            DEBUG_context_expand_location = debug_context_location_in_image(image, x, y, w, h)
            DEBUG_context_with_context_mask = context.clone()
            DEBUG_context_with_context_mask_location = debug_context_location_in_image(image, x, y, w, h)

        if not output_resize_to_target_size:
            # If keep_window_size is enabled and we have max dimensions, use them instead of w,h
            if keep_window_size and max_w is not None and max_h is not None:
                # Dont't forget to consider context_from_mask_extend_factor
                extended_w = int(max_w * context_from_mask_extend_factor)
                extended_h = int(max_h * context_from_mask_extend_factor)
                canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, extended_w, extended_h, output_padding, downscale_algorithm, upscale_algorithm)
            else:
                # Use the actual context dimensions w,h which already include the extend factor
                canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, w, h, output_padding, downscale_algorithm, upscale_algorithm)
        else: # if output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, output_target_width, output_target_height, output_padding, downscale_algorithm, upscale_algorithm)
        
        if self.DEBUG_MODE:
            DEBUG_context_to_target = context.clone()
            DEBUG_context_to_target_location = debug_context_location_in_image(image, x, y, w, h)
            DEBUG_context_to_target_image = image.clone()
            DEBUG_context_to_target_mask = mask.clone()
            DEBUG_canvas_image = canvas_image.clone()
            DEBUG_orig_in_canvas_location = debug_context_location_in_image(canvas_image, cto_x, cto_y, cto_w, cto_h)
            DEBUG_cropped_in_canvas_location = debug_context_location_in_image(canvas_image, ctc_x, ctc_y, ctc_w, ctc_h)

        # For blending, grow the mask even further and make it blurrier.
        cropped_mask_blend = cropped_mask.clone()
        if mask_blend_pixels > 0:
           cropped_mask_blend = blur_m(cropped_mask_blend, mask_blend_pixels*0.5)
        if self.DEBUG_MODE:
            DEBUG_cropped_mask_blend = cropped_mask_blend.clone()

        stitcher = {
            'canvas_to_orig_x': cto_x,
            'canvas_to_orig_y': cto_y,
            'canvas_to_orig_w': cto_w,
            'canvas_to_orig_h': cto_h,
            'canvas_image': canvas_image,
            'cropped_to_canvas_x': ctc_x,
            'cropped_to_canvas_y': ctc_y,
            'cropped_to_canvas_w': ctc_w,
            'cropped_to_canvas_h': ctc_h,
            'cropped_mask_for_blend': cropped_mask_blend,
        }

        if not self.DEBUG_MODE:
            return stitcher, cropped_image, cropped_mask
        else:
            return stitcher, cropped_image, cropped_mask, DEBUG_preresize_image, DEBUG_preresize_mask, DEBUG_fillholes_mask, DEBUG_expand_mask, DEBUG_invert_mask, DEBUG_blur_mask, DEBUG_hipassfilter_mask, DEBUG_extend_image, DEBUG_extend_mask, DEBUG_context_from_mask, DEBUG_context_from_mask_location, DEBUG_context_expand, DEBUG_context_expand_location, DEBUG_context_with_context_mask, DEBUG_context_with_context_mask_location, DEBUG_context_to_target, DEBUG_context_to_target_location, DEBUG_context_to_target_image, DEBUG_context_to_target_mask, DEBUG_canvas_image, DEBUG_orig_in_canvas_location, DEBUG_cropped_in_canvas_location, DEBUG_cropped_mask_blend


    def inpaint_crop_single_image(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, mask, optional_context_mask):
        if preresize:
            image, mask, optional_context_mask = preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height)
        if self.DEBUG_MODE:
            DEBUG_preresize_image = image.clone()
            DEBUG_preresize_mask = mask.clone()
       
        if mask_fill_holes:
           mask = fillholes_iterative_hipass_fill_m(mask)
        if self.DEBUG_MODE:
            DEBUG_fillholes_mask = mask.clone()

        if mask_expand_pixels > 0:
            mask = expand_m(mask, mask_expand_pixels)
        if self.DEBUG_MODE:
            DEBUG_expand_mask = mask.clone()

        if mask_invert:
            mask = invert_m(mask)
        if self.DEBUG_MODE:
            DEBUG_invert_mask = mask.clone()

        if mask_blend_pixels > 0:
            mask = expand_m(mask, mask_blend_pixels)
            mask = blur_m(mask, mask_blend_pixels*0.5)
        if self.DEBUG_MODE:
            DEBUG_blur_mask = mask.clone()

        if mask_hipass_filter >= 0.01:
            mask = hipassfilter_m(mask, mask_hipass_filter)
            optional_context_mask = hipassfilter_m(optional_context_mask, mask_hipass_filter)
        if self.DEBUG_MODE:
            DEBUG_hipassfilter_mask = mask.clone()

        if extend_for_outpainting:
            image, mask, optional_context_mask = extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor)
        if self.DEBUG_MODE:
            DEBUG_extend_image = image.clone()
            DEBUG_extend_mask = mask.clone()

        context, x, y, w, h = findcontextarea_m(mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_from_mask = context.clone()
            DEBUG_context_from_mask_location = debug_context_location_in_image(image, x, y, w, h)

        if context_from_mask_extend_factor >= 1.01:
            context, x, y, w, h = growcontextarea_m(context, mask, x, y, w, h, context_from_mask_extend_factor)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_expand = context.clone()
            DEBUG_context_expand_location = debug_context_location_in_image(image, x, y, w, h)

        context, x, y, w, h = combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_with_context_mask = context.clone()
            DEBUG_context_with_context_mask_location = debug_context_location_in_image(image, x, y, w, h)

        if not output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, w, h, output_padding, downscale_algorithm, upscale_algorithm)
        else: # if output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, output_target_width, output_target_height, output_padding, downscale_algorithm, upscale_algorithm)
        if self.DEBUG_MODE:
            DEBUG_context_to_target = context.clone()
            DEBUG_context_to_target_location = debug_context_location_in_image(image, x, y, w, h)
            DEBUG_context_to_target_image = image.clone()
            DEBUG_context_to_target_mask = mask.clone()
            DEBUG_canvas_image = canvas_image.clone()
            DEBUG_orig_in_canvas_location = debug_context_location_in_image(canvas_image, cto_x, cto_y, cto_w, cto_h)
            DEBUG_cropped_in_canvas_location = debug_context_location_in_image(canvas_image, ctc_x, ctc_y, ctc_w, ctc_h)

        # For blending, grow the mask even further and make it blurrier.
        cropped_mask_blend = cropped_mask.clone()
        if mask_blend_pixels > 0:
           cropped_mask_blend = blur_m(cropped_mask_blend, mask_blend_pixels*0.5)
        if self.DEBUG_MODE:
            DEBUG_cropped_mask_blend = cropped_mask_blend.clone()

        stitcher = {
            'canvas_to_orig_x': cto_x,
            'canvas_to_orig_y': cto_y,
            'canvas_to_orig_w': cto_w,
            'canvas_to_orig_h': cto_h,
            'canvas_image': canvas_image,
            'cropped_to_canvas_x': ctc_x,
            'cropped_to_canvas_y': ctc_y,
            'cropped_to_canvas_w': ctc_w,
            'cropped_to_canvas_h': ctc_h,
            'cropped_mask_for_blend': cropped_mask_blend,
        }

        if not self.DEBUG_MODE:
            return stitcher, cropped_image, cropped_mask
        else:
            return stitcher, cropped_image, cropped_mask, DEBUG_preresize_image, DEBUG_preresize_mask, DEBUG_fillholes_mask, DEBUG_expand_mask, DEBUG_invert_mask, DEBUG_blur_mask, DEBUG_hipassfilter_mask, DEBUG_extend_image, DEBUG_extend_mask, DEBUG_context_from_mask, DEBUG_context_from_mask_location, DEBUG_context_expand, DEBUG_context_expand_location, DEBUG_context_with_context_mask, DEBUG_context_with_context_mask_location, DEBUG_context_to_target, DEBUG_context_to_target_location, DEBUG_context_to_target_image, DEBUG_context_to_target_mask, DEBUG_canvas_image, DEBUG_orig_in_canvas_location, DEBUG_cropped_in_canvas_location, DEBUG_cropped_mask_blend


class InpaintStitchImproved:
    """
    Advanced inpainting stitch node that reconstructs full images from cropped inpainted regions.
    
    This node takes the output from InpaintCropImproved and stitches the inpainted results back 
    into the original image context. It supports multiple blending modes and handles batch processing
    for video sequences while maintaining temporal consistency.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER", {"tooltip": "Stitcher data from InpaintCropImproved containing reconstruction information including canvas coordinates, dimensions, and blending masks."}),
                "inpainted_image": ("IMAGE", {"tooltip": "The inpainted image(s) to be stitched back. Should correspond to the cropped output from InpaintCropImproved after inpainting."}),
                "stitch_mode": (["mask_only", "entire_crop", "blend_entire"], {"default": "mask_only", "tooltip": "Stitching method: 'mask_only' blends only masked areas, 'entire_crop' replaces the entire cropped region, 'blend_entire' smoothly blends the entire crop with surrounding areas."}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of blending operation. 1.0 = full inpainted result, 0.0 = original image preserved. Only affects blend modes."}),
            }
        }

    CATEGORY = "inpaint"
    DESCRIPTION = """
    Reconstructs full-resolution images by stitching inpainted crops back into their original context.
    
    Features:
    - Multiple stitching modes for different use cases
    - Batch processing support for video sequences  
    - Smart blending to avoid seams and artifacts
    - Automatic coordinate transformation and scaling
    - Preserves original image quality in non-inpainted areas
    
    The node handles all the complex coordinate transformations, scaling, and blending
    needed to seamlessly integrate inpainted content back into the source images.
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Reconstructed full-resolution image(s) with inpainted content seamlessly integrated. Maintains original dimensions and quality.",)

    FUNCTION = "inpaint_stitch"

    def inpaint_stitch(self, stitcher, inpainted_image, stitch_mode, blend_strength):
        inpainted_image = inpainted_image.clone()
        results = []

        batch_size = inpainted_image.shape[0]
        assert len(stitcher['cropped_to_canvas_x']) == batch_size or len(stitcher['cropped_to_canvas_x']) == 1, "Stitch batch size doesn't match image batch size"
        override = False
        if len(stitcher['cropped_to_canvas_x']) != batch_size and len(stitcher['cropped_to_canvas_x']) == 1:
            override = True
        for b in range(batch_size):
            one_image = inpainted_image[b]
            one_stitcher = {}
            for key in ['downscale_algorithm', 'upscale_algorithm', 'blend_pixels']:
                one_stitcher[key] = stitcher[key]
            for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                if override: # One stitcher for many images, always read 0.
                    one_stitcher[key] = stitcher[key][0]
                else:
                    one_stitcher[key] = stitcher[key][b]
            one_image = one_image.unsqueeze(0)
            one_image, = self.inpaint_stitch_single_image(one_stitcher, one_image, stitch_mode, blend_strength)
            one_image = one_image.squeeze(0)
            one_image = one_image.clone()
            results.append(one_image)

        result_batch = torch.stack(results, dim=0)

        return (result_batch,)

    def inpaint_stitch_single_image(self, stitcher, inpainted_image, stitch_mode, blend_strength):
        downscale_algorithm = stitcher['downscale_algorithm']
        upscale_algorithm = stitcher['upscale_algorithm']
        canvas_image = stitcher['canvas_image']

        ctc_x = stitcher['cropped_to_canvas_x']
        ctc_y = stitcher['cropped_to_canvas_y']
        ctc_w = stitcher['cropped_to_canvas_w']
        ctc_h = stitcher['cropped_to_canvas_h']

        cto_x = stitcher['canvas_to_orig_x']
        cto_y = stitcher['canvas_to_orig_y']
        cto_w = stitcher['canvas_to_orig_w']
        cto_h = stitcher['canvas_to_orig_h']

        mask = stitcher['cropped_mask_for_blend']  # shape: [1, H, W]

        if stitch_mode == "mask_only":
            # Original behavior - use mask for blending
            output_image = stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm)
        
        elif stitch_mode == "entire_crop":
            # Replace entire cropped region - create a full mask
            full_mask = torch.ones_like(mask)
            output_image = stitch_magic_im(canvas_image, inpainted_image, full_mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm)
        
        elif stitch_mode == "blend_entire":
            # Blend entire cropped region with adjustable strength
            full_mask = torch.ones_like(mask) * blend_strength
            output_image = stitch_magic_im(canvas_image, inpainted_image, full_mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm)

        return (output_image,)
