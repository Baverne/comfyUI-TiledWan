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


class Tile:
    """
    Represents a single tile with its content and position in the tiling grid
    """
    def __init__(self, content, temporal_index, line_index, column_index, 
                 temporal_range, spatial_range_h, spatial_range_w):
        self.content = content  # The actual tile tensor
        self.temporal_index = temporal_index  # Index in temporal dimension
        self.line_index = line_index  # Index in height/line dimension
        self.column_index = column_index  # Index in width/column dimension
        
        # Store the ranges for overlap calculations
        self.temporal_range = temporal_range  # (start_frame, end_frame)
        self.spatial_range_h = spatial_range_h  # (start_h, end_h)
        self.spatial_range_w = spatial_range_w  # (start_w, end_w)
    
    def __str__(self):
        return f"Tile[T{self.temporal_index}:L{self.line_index}:C{self.column_index}] " \
               f"Shape: {self.content.shape} " \
               f"Temporal: {self.temporal_range} " \
               f"Spatial: H{self.spatial_range_h} W{self.spatial_range_w}"


class TileAndStitchBack:
    """
    Tile and Stitch Back node with proper dimension-wise stitching.
    
    New approach:
    1. Temporal tiling: Split video into chunks with overlap
    2. Spatial tiling: Split each chunk into tiles with overlap  
    3. Processing: Apply transformations to each tile
    4. Dimension-wise stitching:
       a) Column-wise: Stitch tiles in same column (same temporal chunk)
       b) Line-wise: Stitch columns together (complete temporal chunks)
       c) Time-wise: Stitch temporal chunks together
    
    This approach eliminates the black-to-color ramp issue by handling
    overlaps systematically in each dimension.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Tile and Stitch Back inputs
        """
        return {
            "required": {
                "video": ("IMAGE",),
                "target_frames": ("INT", {"default": 81, "min": 16, "max": 200, "step": 1}),
                "target_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "target_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "frame_overlap": ("INT", {"default": 10, "min": 0, "max": 40, "step": 1}),
                "spatial_overlap": ("INT", {"default": 20, "min": 0, "max": 100, "step": 4}),
                "color_shift_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "debug_mode": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("stitched_video", "tile_info")
    FUNCTION = "tile_and_stitch"
    OUTPUT_NODE = True
    CATEGORY = "TiledWan"

    def tile_and_stitch(self, video, target_frames, target_width, target_height, 
                       frame_overlap, spatial_overlap, color_shift_strength, debug_mode):
        """
        Tile video temporally and spatially, apply transformations, then stitch back using dimension-wise approach
        """
        
        print("\n" + "="*80)
        print("                  TILE AND STITCH BACK (NEW ALGORITHM)")
        print("="*80)
        print("🧩 Starting dimension-wise video tiling and stitching...")
        
        try:
            import random
            import traceback
            
            # Input video info
            batch_size, height, width, channels = video.shape
            print(f"📹 Input video shape: {video.shape} (B×H×W×C)")
            print(f"🎯 Target tile size: {target_frames} frames × {target_width}×{target_height}")
            print(f"🔗 Overlaps: {frame_overlap} frames, {spatial_overlap} pixels")
            
            # Calculate tile ranges
            temporal_tiles = self._calculate_temporal_tiles(batch_size, target_frames, frame_overlap)
            spatial_tiles_h = self._calculate_spatial_tiles(height, target_height, spatial_overlap)
            spatial_tiles_w = self._calculate_spatial_tiles(width, target_width, spatial_overlap)
            
            print(f"⏱️  Temporal chunks: {len(temporal_tiles)}")
            print(f"🗺️  Spatial grid: {len(spatial_tiles_h)}×{len(spatial_tiles_w)} tiles per frame")
            print(f"📦 Total tiles: {len(temporal_tiles) * len(spatial_tiles_h) * len(spatial_tiles_w)}")
            
            # STEP 1: Extract and process all tiles
            print(f"\n🔸 STEP 1: Extracting and processing tiles...")
            all_tiles = self._extract_and_process_tiles(
                video, temporal_tiles, spatial_tiles_h, spatial_tiles_w,
                color_shift_strength, debug_mode
            )
            
            # STEP 2: Column-wise stitching (stitch tiles in same column, same temporal chunk)
            print(f"\n🔸 STEP 2: Column-wise stitching...")
            column_strips = self._stitch_columns(all_tiles, spatial_overlap, debug_mode)
            
            # STEP 3: Line-wise stitching (stitch column strips to complete temporal chunks)
            print(f"\n🔸 STEP 3: Line-wise stitching...")
            temporal_chunks = self._stitch_lines(column_strips, spatial_overlap, debug_mode)
            
            # STEP 4: Time-wise stitching (stitch temporal chunks together)
            print(f"\n🔸 STEP 4: Time-wise stitching...")
            stitched_video = self._stitch_temporal_chunks_new(temporal_chunks, temporal_tiles, frame_overlap)
            
            # STEP 5: Crop back to original dimensions
            print(f"\n🔸 STEP 5: Cropping to original dimensions...")
            print(f"   📐 Stitched video shape: {stitched_video.shape}")
            print(f"   🎯 Target shape: [{batch_size}, {height}, {width}, {channels}]")
            
            # Crop spatial dimensions back to original size
            final_video = stitched_video[:batch_size, :height, :width, :channels]
            
            print(f"   ✂️  Cropped to: {final_video.shape}")
            
            # Generate summary
            tile_info_summary = self._generate_tile_info_summary_new(
                all_tiles, temporal_tiles, spatial_tiles_h, spatial_tiles_w
            )
            
            print(f"✅ Dimension-wise tiling and stitching completed!")
            print(f"📤 Final video shape: {final_video.shape} (cropped to match input)")
            print(f"🧩 Total tiles processed: {len(all_tiles)}")
            print(f"📏 Original input: {batch_size}×{height}×{width}×{channels}")
            print(f"📏 After stitching: {stitched_video.shape}")
            print(f"📏 After cropping: {final_video.shape}")
            print("="*80 + "\n")
            
            return (final_video, tile_info_summary)
            
        except Exception as e:
            print(f"❌ Error in tile and stitch: {str(e)}")
            print(f"📋 Full traceback:")
            print(traceback.format_exc())
            print("="*80 + "\n")
            
            # Return original video in case of error
            dummy_info = f"Error during tiling: {str(e)}"
            return (video, dummy_info)
    
    def _extract_and_process_tiles(self, video, temporal_tiles, spatial_tiles_h, spatial_tiles_w,
                                  color_shift_strength, debug_mode):
        """
        Extract all tiles and apply transformations, storing them in Tile objects
        """
        all_tiles = []
        
        for temporal_idx, (t_start, t_end) in enumerate(temporal_tiles):
            if debug_mode:
                print(f"   🎬 Temporal chunk {temporal_idx + 1}/{len(temporal_tiles)} (frames {t_start}-{t_end-1})")
            
            # Extract temporal chunk
            chunk = video[t_start:t_end]
            
            # Process each spatial tile within this temporal chunk
            for h_idx, (h_start, h_end) in enumerate(spatial_tiles_h):
                for w_idx, (w_start, w_end) in enumerate(spatial_tiles_w):
                    if debug_mode and temporal_idx == 0:  # Only show spatial info for first temporal chunk
                        print(f"      🧩 Spatial tile L{h_idx}:C{w_idx} H[{h_start}:{h_end}] × W[{w_start}:{w_end}]")
                    
                    # Extract spatial tile
                    tile_content = chunk[:, h_start:h_end, w_start:w_end, :]
                    
                    # Apply color transformation (for debugging)
                    tile_transformed = self._apply_color_shift(tile_content, color_shift_strength, 
                                                             temporal_idx, h_idx, w_idx)
                    
                    # Create Tile object
                    tile = Tile(
                        content=tile_transformed,
                        temporal_index=temporal_idx,
                        line_index=h_idx,
                        column_index=w_idx,
                        temporal_range=(t_start, t_end),
                        spatial_range_h=(h_start, h_end),
                        spatial_range_w=(w_start, w_end)
                    )
                    
                    all_tiles.append(tile)
        
        print(f"   ✅ Extracted {len(all_tiles)} tiles")
        return all_tiles
    
    def _stitch_columns(self, all_tiles, spatial_overlap, debug_mode):
        """
        Stitch tiles vertically (same column, same temporal chunk) to create column strips
        """
        # Group tiles by temporal_index and column_index
        column_groups = {}
        for tile in all_tiles:
            key = (tile.temporal_index, tile.column_index)
            if key not in column_groups:
                column_groups[key] = []
            column_groups[key].append(tile)
        
        # Sort each group by line_index
        for key in column_groups:
            column_groups[key].sort(key=lambda t: t.line_index)
        
        column_strips = []
        
        for (temporal_idx, col_idx), column_tiles in column_groups.items():
            if debug_mode and temporal_idx == 0:  # Only show info for first temporal chunk
                print(f"   📏 Column {col_idx}: {len(column_tiles)} tiles")
            
            # Stitch tiles in this column vertically
            if len(column_tiles) == 1:
                # Single tile - no stitching needed
                column_strip = column_tiles[0].content
            else:
                # Multiple tiles - stitch vertically with overlap
                column_strip = self._stitch_tiles_vertically(column_tiles, spatial_overlap)
            
            # Store column strip with its position info
            strip_info = {
                'content': column_strip,
                'temporal_index': temporal_idx,
                'column_index': col_idx,
                'spatial_range_w': column_tiles[0].spatial_range_w,  # All tiles in column have same W range
                'spatial_range_h': (column_tiles[0].spatial_range_h[0], column_tiles[-1].spatial_range_h[1])
            }
            column_strips.append(strip_info)
        
        print(f"   ✅ Created {len(column_strips)} column strips")
        return column_strips
    
    def _stitch_lines(self, column_strips, spatial_overlap, debug_mode):
        """
        Stitch column strips horizontally (same temporal chunk) to create complete temporal chunks
        """
        # Group column strips by temporal_index
        temporal_groups = {}
        for strip in column_strips:
            temporal_idx = strip['temporal_index']
            if temporal_idx not in temporal_groups:
                temporal_groups[temporal_idx] = []
            temporal_groups[temporal_idx].append(strip)
        
        # Sort each group by column_index
        for temporal_idx in temporal_groups:
            temporal_groups[temporal_idx].sort(key=lambda s: s['column_index'])
        
        temporal_chunks = []
        
        for temporal_idx, strips in temporal_groups.items():
            if debug_mode and temporal_idx == 0:  # Only show info for first temporal chunk
                print(f"   📐 Temporal chunk {temporal_idx}: {len(strips)} column strips")
            
            # Stitch column strips horizontally
            if len(strips) == 1:
                # Single strip - no stitching needed
                chunk_content = strips[0]['content']
            else:
                # Multiple strips - stitch horizontally with overlap
                chunk_content = self._stitch_strips_horizontally(strips, spatial_overlap)
            
            # Store temporal chunk with its info
            chunk_info = {
                'content': chunk_content,
                'temporal_index': temporal_idx,
                'spatial_range_h': strips[0]['spatial_range_h'],  # All strips have same H range
                'spatial_range_w': (strips[0]['spatial_range_w'][0], strips[-1]['spatial_range_w'][1])
            }
            temporal_chunks.append(chunk_info)
        
        # Sort by temporal index
        temporal_chunks.sort(key=lambda c: c['temporal_index'])
        
        print(f"   ✅ Created {len(temporal_chunks)} temporal chunks")
        return temporal_chunks
    
    def _stitch_tiles_vertically(self, column_tiles, spatial_overlap):
        """
        Stitch tiles vertically (in height dimension) with proper fade blending
        """
        if len(column_tiles) == 1:
            return column_tiles[0].content
        
        # Get dimensions
        first_tile = column_tiles[0].content
        total_height = sum(tile.spatial_range_h[1] - tile.spatial_range_h[0] for tile in column_tiles)
        # Subtract overlaps
        total_height -= spatial_overlap * (len(column_tiles) - 1)
        
        width = first_tile.shape[2]
        channels = first_tile.shape[3]
        frames = first_tile.shape[0]
        
        # Create result tensor
        result = torch.zeros((frames, total_height, width, channels), 
                           dtype=first_tile.dtype, device=first_tile.device)
        
        current_h = 0
        for i, tile in enumerate(column_tiles):
            tile_h = tile.content.shape[1]
            
            if i == 0:
                # First tile - place directly
                result[:, current_h:current_h+tile_h, :, :] = tile.content
                current_h += tile_h
            else:
                # Subsequent tiles - check for last tile with large overlap
                is_last_tile = (i == len(column_tiles) - 1)
                
                # Calculate actual overlap by comparing tile positions
                expected_start = tile.spatial_range_h[0]
                actual_overlap = current_h - expected_start
                actual_overlap = max(0, actual_overlap)  # Ensure non-negative
                
                if is_last_tile and actual_overlap > spatial_overlap:
                    # Last tile with large overlap - fade across entire overlapping region
                    print(f"      🔥 Last tile in column: Large overlap detected ({actual_overlap} > {spatial_overlap})")
                    overlap_start = expected_start
                    overlap_size = actual_overlap
                    non_overlap_start = current_h
                    non_overlap_size = tile_h - actual_overlap
                    
                    # Place non-overlapping part
                    if non_overlap_size > 0:
                        result[:, non_overlap_start:non_overlap_start+non_overlap_size, :, :] = \
                            tile.content[:, actual_overlap:, :, :]
                    
                    # Handle large overlapping region with fade blending
                    if overlap_size > 0:
                        existing_region = result[:, overlap_start:overlap_start+overlap_size, :, :]
                        new_region = tile.content[:, :overlap_size, :, :]
                        
                        # Create fade mask for entire overlapping region
                        fade_mask = self._create_vertical_fade_mask(overlap_size, existing_region.dtype, existing_region.device)
                        
                        # Apply fade blending
                        blended = existing_region * (1.0 - fade_mask) + new_region * fade_mask
                        result[:, overlap_start:overlap_start+overlap_size, :, :] = blended
                    
                    current_h += non_overlap_size
                else:
                    # Normal tile with standard overlap handling
                    overlap_start = current_h - spatial_overlap
                    
                    # Place non-overlapping part first
                    non_overlap_h = tile_h - spatial_overlap
                    result[:, current_h:current_h+non_overlap_h, :, :] = tile.content[:, spatial_overlap:, :, :]
                    
                    # Handle overlapping region with fade blending
                    if spatial_overlap > 0:
                        existing_region = result[:, overlap_start:current_h, :, :]
                        new_region = tile.content[:, :spatial_overlap, :, :]
                        
                        # Create vertical fade mask
                        fade_mask = self._create_vertical_fade_mask(spatial_overlap, existing_region.dtype, existing_region.device)
                        
                        # Apply fade blending
                        blended = existing_region * (1.0 - fade_mask) + new_region * fade_mask
                        result[:, overlap_start:current_h, :, :] = blended
                    
                    current_h += non_overlap_h
        
        return result
    
    def _stitch_strips_horizontally(self, strips, spatial_overlap):
        """
        Stitch column strips horizontally (in width dimension) with proper fade blending
        """
        if len(strips) == 1:
            return strips[0]['content']
        
        # Get dimensions
        first_strip = strips[0]['content']
        total_width = sum(s['spatial_range_w'][1] - s['spatial_range_w'][0] for s in strips)
        # Subtract overlaps
        total_width -= spatial_overlap * (len(strips) - 1)
        
        height = first_strip.shape[1]
        channels = first_strip.shape[3]
        frames = first_strip.shape[0]
        
        # Create result tensor
        result = torch.zeros((frames, height, total_width, channels), 
                           dtype=first_strip.dtype, device=first_strip.device)
        
        current_w = 0
        for i, strip in enumerate(strips):
            strip_content = strip['content']
            strip_w = strip_content.shape[2]
            
            if i == 0:
                # First strip - place directly
                result[:, :, current_w:current_w+strip_w, :] = strip_content
                current_w += strip_w
            else:
                # Subsequent strips - check for last strip with large overlap
                is_last_strip = (i == len(strips) - 1)
                
                # Calculate actual overlap by comparing strip positions
                expected_start = strip['spatial_range_w'][0]
                actual_overlap = current_w - expected_start
                actual_overlap = max(0, actual_overlap)  # Ensure non-negative
                
                if is_last_strip and actual_overlap > spatial_overlap:
                    # Last strip with large overlap - fade across entire overlapping region
                    print(f"      🔥 Last strip in line: Large overlap detected ({actual_overlap} > {spatial_overlap})")
                    overlap_start = expected_start
                    overlap_size = actual_overlap
                    non_overlap_start = current_w
                    non_overlap_size = strip_w - actual_overlap
                    
                    # Place non-overlapping part
                    if non_overlap_size > 0:
                        result[:, :, non_overlap_start:non_overlap_start+non_overlap_size, :] = \
                            strip_content[:, :, actual_overlap:, :]
                    
                    # Handle large overlapping region with fade blending
                    if overlap_size > 0:
                        existing_region = result[:, :, overlap_start:overlap_start+overlap_size, :]
                        new_region = strip_content[:, :, :overlap_size, :]
                        
                        # Create fade mask for entire overlapping region
                        fade_mask = self._create_horizontal_fade_mask(overlap_size, existing_region.dtype, existing_region.device)
                        
                        # Apply fade blending
                        blended_region = existing_region * (1 - fade_mask) + new_region * fade_mask
                        result[:, :, overlap_start:overlap_start+overlap_size, :] = blended_region
                    
                    current_w += non_overlap_size
                else:
                    # Normal strip with standard overlap handling
                    overlap_start = current_w - spatial_overlap
                    
                    # Place non-overlapping part first
                    non_overlap_w = strip_w - spatial_overlap
                    result[:, :, current_w:current_w+non_overlap_w, :] = strip_content[:, :, spatial_overlap:, :]
                    
                    # Handle overlapping region with fade blending
                    if spatial_overlap > 0:
                        existing_region = result[:, :, overlap_start:current_w, :]
                        new_region = strip_content[:, :, :spatial_overlap, :]
                        
                        # Create horizontal fade mask
                        fade_mask = self._create_horizontal_fade_mask(spatial_overlap, existing_region.dtype, existing_region.device)
                        
                        # Apply fade blending
                        blended_region = existing_region * (1 - fade_mask) + new_region * fade_mask
                        result[:, :, overlap_start:current_w, :] = blended_region
                    
                    current_w += non_overlap_w
            
        return result
            
        return result
    
    def _create_vertical_fade_mask(self, fade_size, dtype, device):
        """Create a vertical fade mask for blending tiles vertically."""
        fade_mask = torch.linspace(0, 1, fade_size, dtype=dtype, device=device)
        # Shape: (1, fade_size, 1, 1) for broadcasting
        return fade_mask.view(1, fade_size, 1, 1)
    
    def _create_horizontal_fade_mask(self, fade_size, dtype, device):
        """Create a horizontal fade mask for blending tiles horizontally."""
        fade_mask = torch.linspace(0, 1, fade_size, dtype=dtype, device=device)
        # Shape: (1, 1, fade_size, 1) for broadcasting
        return fade_mask.view(1, 1, fade_size, 1)
    
    def _stitch_temporal_chunks_new(self, temporal_chunks, temporal_tiles, frame_overlap):
        """Stitch temporal chunks to create the final output with temporal blending."""
        if not temporal_chunks:
            return None
            
        if len(temporal_chunks) == 1:
            return temporal_chunks[0]['content']
        
        # Get dimensions from first chunk
        first_chunk_content = temporal_chunks[0]['content']
        
        # Calculate total frames accounting for overlaps
        total_frames = sum(chunk['content'].shape[0] for chunk in temporal_chunks)
        if frame_overlap > 0:
            total_frames -= (len(temporal_chunks) - 1) * frame_overlap
        
        # Get spatial dimensions from the stitched chunks (not original input)
        height = first_chunk_content.shape[1]
        width = first_chunk_content.shape[2]
        channels = first_chunk_content.shape[3]
        
        print(f"   📏 Temporal stitching dimensions:")
        print(f"      • Total frames: {total_frames}")
        print(f"      • Spatial size: {height}×{width}")
        print(f"      • Channels: {channels}")
        print(f"      • Chunk sizes: {[chunk['content'].shape for chunk in temporal_chunks]}")
        
        # Initialize result tensor with correct dimensions
        result = torch.zeros(
            (total_frames, height, width, channels),
            dtype=first_chunk_content.dtype,
            device=first_chunk_content.device
        )
        
        current_t = 0
        for i, chunk_info in enumerate(temporal_chunks):
            chunk_content = chunk_info['content']
            chunk_frames = chunk_content.shape[0]
            
            if i == 0:
                # First chunk - place entirely
                result[current_t:current_t+chunk_frames] = chunk_content
                current_t += chunk_frames
            else:
                # Subsequent chunks - check for last chunk with large overlap
                is_last_chunk = (i == len(temporal_chunks) - 1)
                
                # Get the temporal range for this chunk from temporal_tiles
                temporal_tile_info = temporal_tiles[i]
                expected_start = temporal_tile_info[0]
                actual_overlap = current_t - expected_start
                actual_overlap = max(0, actual_overlap)  # Ensure non-negative
                
                if is_last_chunk and actual_overlap > frame_overlap:
                    # Last chunk with large overlap - fade across entire overlapping region
                    print(f"      🔥 Last temporal chunk: Large overlap detected ({actual_overlap} > {frame_overlap})")
                    overlap_start = expected_start
                    overlap_size = actual_overlap
                    non_overlap_start = current_t
                    non_overlap_size = chunk_frames - actual_overlap
                    
                    # Place non-overlapping part
                    if non_overlap_size > 0:
                        result[non_overlap_start:non_overlap_start+non_overlap_size] = \
                            chunk_content[actual_overlap:]
                    
                    # Handle large overlapping region with fade blending
                    if overlap_size > 0:
                        existing_region = result[overlap_start:overlap_start+overlap_size]
                        new_region = chunk_content[:overlap_size]
                        
                        # Create temporal fade mask for entire overlapping region
                        fade_mask = self._create_temporal_fade_mask(overlap_size, existing_region.dtype, existing_region.device)
                        
                        # Apply fade blending
                        blended_region = existing_region * (1 - fade_mask) + new_region * fade_mask
                        result[overlap_start:overlap_start+overlap_size] = blended_region
                    
                    current_t += non_overlap_size
                else:
                    # Normal chunk with standard overlap handling
                    non_overlap_frames = chunk_frames - frame_overlap
                    overlap_start = current_t - frame_overlap
                    
                    # Place non-overlapping part
                    result[current_t:current_t+non_overlap_frames] = chunk_content[frame_overlap:]
                    
                    # Handle overlapping region with temporal fade blending
                    if frame_overlap > 0:
                        existing_region = result[overlap_start:current_t]
                        new_region = chunk_content[:frame_overlap]
                        
                        # Create temporal fade mask
                        fade_mask = self._create_temporal_fade_mask(frame_overlap, existing_region.dtype, existing_region.device)
                        
                        # Apply fade blending
                        blended_region = existing_region * (1 - fade_mask) + new_region * fade_mask
                        result[overlap_start:current_t] = blended_region
                    
                    current_t += non_overlap_frames
        
        return result
    
    def _create_temporal_fade_mask(self, fade_size, dtype, device):
        """Create a temporal fade mask for blending across time dimension."""
        fade_mask = torch.linspace(0, 1, fade_size, dtype=dtype, device=device)
        # Shape: (fade_size, 1, 1, 1) for broadcasting
        return fade_mask.view(fade_size, 1, 1, 1)
    
    def _generate_tile_info_summary_new(self, tiles, temporal_tiles, spatial_tiles_h, spatial_tiles_w):
        """Generate a summary of tile processing for the new dimension-wise algorithm."""
        summary_lines = []
        summary_lines.append("=== Tile Processing Summary (Dimension-wise Algorithm) ===")
        summary_lines.append(f"Input video - Temporal chunks: {len(temporal_tiles)}")
        summary_lines.append(f"Input video - Spatial grid: {len(spatial_tiles_h)}×{len(spatial_tiles_w)}")
        summary_lines.append(f"Total tiles processed: {len(tiles)}")
        
        # Group tiles by temporal chunk
        temporal_chunks = {}
        for tile in tiles:
            t_idx = tile.temporal_index
            if t_idx not in temporal_chunks:
                temporal_chunks[t_idx] = []
            temporal_chunks[t_idx].append(tile)
        
        summary_lines.append(f"Temporal chunks: {len(temporal_chunks)}")
        
        # Analyze grid structure
        for t_idx, chunk_tiles in temporal_chunks.items():
            line_groups = {}
            for tile in chunk_tiles:
                l_idx = tile.line_index
                if l_idx not in line_groups:
                    line_groups[l_idx] = []
                line_groups[l_idx].append(tile)
            
            max_cols = max(len(line_tiles) for line_tiles in line_groups.values())
            summary_lines.append(f"  Chunk {t_idx}: {len(line_groups)} lines × {max_cols} columns")
        
        summary_lines.append("Processing order: Column-wise → Line-wise → Temporal → Crop")
        summary_lines.append("Last tiles in each dimension use full-region fade blending")
        summary_lines.append("Final video cropped back to original input dimensions")
        summary_lines.append("="*60)
        
        return "\n".join(summary_lines)
    
    def _calculate_temporal_tiles(self, total_frames, target_frames, overlap):
        """Calculate temporal tile ranges with overlap handling"""
        tiles = []
        
        if total_frames <= target_frames:
            # If video is shorter than target, use the whole video
            tiles.append((0, total_frames))
        else:
            stride = target_frames - overlap
            current = 0
            
            while current < total_frames:
                end = min(current + target_frames, total_frames)
                
                # If this would be the last tile and it doesn't cover remaining frames
                remaining = total_frames - end
                if remaining > 0:
                    # Add this tile normally
                    tiles.append((current, end))
                    # Check if we need another tile to cover the remaining frames
                    if remaining < stride:
                        # Add a final tile that is exactly target_frames but starts from the end
                        final_start = total_frames - target_frames
                        tiles.append((final_start, total_frames))
                        break
                else:
                    # This tile reaches exactly to the end
                    tiles.append((current, end))
                    break
                    
                current += stride
        
        return tiles
    
    def _calculate_spatial_tiles(self, total_size, target_size, overlap):
        """Calculate spatial tile ranges with overlap handling"""
        tiles = []
        
        if total_size <= target_size:
            # If dimension is smaller than target, use the whole dimension
            tiles.append((0, total_size))
        else:
            stride = target_size - overlap
            current = 0
            
            while current < total_size:
                end = min(current + target_size, total_size)
                
                # If this would be the last tile and it doesn't cover remaining pixels
                remaining = total_size - end
                if remaining > 0:
                    # Add this tile normally
                    tiles.append((current, end))
                    # Check if we need another tile to cover the remaining pixels
                    if remaining < stride:
                        # Add a final tile that is exactly target_size but starts from the end
                        final_start = total_size - target_size
                        tiles.append((final_start, total_size))
                        break
                else:
                    # This tile reaches exactly to the end
                    tiles.append((current, end))
                    break
                    
                current += stride
        
        return tiles
    
    def _apply_color_shift(self, tile, strength, temporal_idx, h_idx, w_idx):
        """Apply color transformation to a tile for debugging purposes"""
        if strength <= 0:
            return tile.clone()
        
        # Generate deterministic but varied color shifts based on tile indices
        random.seed(temporal_idx * 1000 + h_idx * 100 + w_idx)
        
        # Random RGB shifts
        r_shift = (random.random() - 0.5) * 2 * strength
        g_shift = (random.random() - 0.5) * 2 * strength  
        b_shift = (random.random() - 0.5) * 2 * strength
        
        transformed = tile.clone()
        
        if tile.shape[-1] >= 3:  # RGB channels
            transformed[:, :, :, 0] = torch.clamp(transformed[:, :, :, 0] + r_shift, 0, 1)
            transformed[:, :, :, 1] = torch.clamp(transformed[:, :, :, 1] + g_shift, 0, 1)
            transformed[:, :, :, 2] = torch.clamp(transformed[:, :, :, 2] + b_shift, 0, 1)
            
            # Debug output to confirm color shift is being applied
            if strength > 0:
                print(f"      🎨 Color shift applied: R{r_shift:+.3f}, G{g_shift:+.3f}, B{b_shift:+.3f}")
        
        return transformed
    
    def _place_tile_with_overlap(self, target, tile, h_start, h_end, w_start, w_end, spatial_overlap):
        """Place tile into target tensor, handling overlaps with spatial fade blending"""
        tile_h, tile_w = tile.shape[1:3]
        
        # Get the target region
        target_region = target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :]
        
        # Check if there's existing content (non-zero) in the target area
        if target_region.sum() > 0:
            # Always use production-quality fade blending regardless of color shift strength
            # Color shift is purely for visual debugging and shouldn't affect blending quality
            fade_mask = self._create_spatial_fade_mask(tile_h, tile_w, target_region, tile, spatial_overlap)
            
            # Apply fade blending: existing * (1 - fade_mask) + tile * fade_mask
            blended = target_region * (1.0 - fade_mask) + tile * fade_mask
            target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :] = blended
        else:
            # First tile in this area or non-overlapping area - always place the tile
            # (this includes color-shifted tiles when debugging is enabled)
            target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :] = tile
    
    def _create_spatial_fade_mask(self, tile_h, tile_w, existing, new_tile, spatial_overlap):
        """Create a spatial fade mask for smooth blending between tiles"""
        # Create base mask (1.0 = use new tile, 0.0 = use existing)
        mask = torch.ones(1, tile_h, tile_w, 1, dtype=existing.dtype, device=existing.device)
        
        # Define fade distance using spatial_overlap parameter
        fade_h = min(spatial_overlap, tile_h // 2)  # Fade over spatial_overlap pixels or half tile height
        fade_w = min(spatial_overlap, tile_w // 2)  # Fade over spatial_overlap pixels or half tile width
        
        # Check which edges have existing content to determine fade direction
        has_top = existing[:, :fade_h, :, :].sum() > 0
        has_bottom = existing[:, -fade_h:, :, :].sum() > 0
        has_left = existing[:, :, :fade_w, :].sum() > 0
        has_right = existing[:, :, -fade_w:, :].sum() > 0
        
        # Create fade gradients for each edge that has existing content
        if has_top:
            # Fade from 0 at top to 1 after fade_h pixels
            for i in range(fade_h):
                alpha = i / fade_h
                mask[:, i, :, :] = alpha
        
        if has_bottom:
            # Fade from 1 before last fade_h pixels to 0 at bottom
            for i in range(fade_h):
                alpha = 1.0 - (i / fade_h)
                mask[:, tile_h - 1 - i, :, :] = alpha
        
        if has_left:
            # Fade from 0 at left to 1 after fade_w pixels
            for i in range(fade_w):
                alpha = i / fade_w
                mask[:, :, i, :] = torch.minimum(mask[:, :, i, :], torch.tensor(alpha, dtype=mask.dtype, device=mask.device))
        
        if has_right:
            # Fade from 1 before last fade_w pixels to 0 at right
            for i in range(fade_w):
                alpha = 1.0 - (i / fade_w)
                mask[:, :, tile_w - 1 - i, :] = torch.minimum(mask[:, :, tile_w - 1 - i, :], torch.tensor(alpha, dtype=mask.dtype, device=mask.device))
        
        # Broadcast mask to match tile dimensions
        return mask.expand_as(new_tile)
    
    def _stitch_temporal_chunks(self, chunks, temporal_tiles, total_frames, height, width, channels, overlap):
        """Stitch temporal chunks back together with temporal fade blending"""
        result = torch.zeros((total_frames, height, width, channels), dtype=chunks[0].dtype, device=chunks[0].device)
        
        for i, ((t_start, t_end), chunk) in enumerate(zip(temporal_tiles, chunks)):
            chunk_frames = chunk.shape[0]
            
            if i == 0:
                # First chunk, place directly
                result[t_start:t_start+chunk_frames] = chunk
            else:
                # Handle overlap with previous chunk using temporal fade
                prev_end = temporal_tiles[i-1][1]
                overlap_start = max(t_start, prev_end - overlap)
                overlap_end = min(t_end, prev_end)
                
                if overlap_start < overlap_end:
                    # There's an overlap, apply temporal fade blending
                    overlap_frames = overlap_end - overlap_start
                    chunk_overlap_start = overlap_start - t_start
                    
                    # Get overlapping regions
                    existing_frames = result[overlap_start:overlap_end]
                    new_frames = chunk[chunk_overlap_start:chunk_overlap_start+overlap_frames]
                    
                    # Create temporal fade mask
                    fade_mask = self._create_temporal_fade_mask(overlap_frames, existing_frames.dtype, existing_frames.device)
                    
                    # Apply temporal fade: existing * (1 - fade) + new * fade
                    blended_frames = existing_frames * (1.0 - fade_mask) + new_frames * fade_mask
                    result[overlap_start:overlap_end] = blended_frames
                    
                    # Place non-overlapping frames
                    if overlap_end < t_start + chunk_frames:
                        non_overlap_start = overlap_end
                        chunk_offset = non_overlap_start - t_start
                        result[non_overlap_start:t_start+chunk_frames] = chunk[chunk_offset:]
                else:
                    # No overlap, place directly
                    result[t_start:t_start+chunk_frames] = chunk
        
        return result
    
    def _create_temporal_fade_mask(self, overlap_frames, dtype, device):
        """Create a temporal fade mask for smooth frame transitions"""
        # Create fade mask: 0.0 at start (keep existing) to 1.0 at end (use new)
        fade_values = torch.linspace(0.0, 1.0, overlap_frames, dtype=dtype, device=device)
        
        # Reshape to broadcast properly: [frames, 1, 1, 1]
        fade_mask = fade_values.view(overlap_frames, 1, 1, 1)
        
        return fade_mask
    
    def _generate_tile_info_summary(self, tile_info_list, temporal_tiles, spatial_tiles_h, spatial_tiles_w):
        """Generate a summary of the tiling process"""
        summary = f"=== TILE AND STITCH SUMMARY ===\n"
        summary += f"Total tiles processed: {len(tile_info_list)}\n"
        summary += f"Temporal chunks: {len(temporal_tiles)}\n"
        summary += f"Spatial tiles per frame: {len(spatial_tiles_h)}×{len(spatial_tiles_w)}\n\n"
        
        summary += "Temporal chunks:\n"
        for i, (start, end) in enumerate(temporal_tiles):
            summary += f"  Chunk {i+1}: frames {start}-{end-1} ({end-start} frames)\n"
        
        summary += f"\nSpatial tiles (Height):\n"
        for i, (start, end) in enumerate(spatial_tiles_h):
            summary += f"  Row {i+1}: pixels {start}-{end-1} ({end-start} pixels)\n"
        
        summary += f"\nSpatial tiles (Width):\n"
        for i, (start, end) in enumerate(spatial_tiles_w):
            summary += f"  Col {i+1}: pixels {start}-{end-1} ({end-start} pixels)\n"
        
        summary += f"\nProcessing completed successfully!"
        
        return summary


class TiledWanVideoVACEpipe:
    """
    Dimension-wise Tiled WanVideo VACE Pipeline - The ultimate node for processing large videos.
    
    This node combines the sophisticated dimension-wise tiling system from TileAndStitchBack with 
    the complete WanVideo VACE pipeline. It handles large videos using the advanced algorithm:
    
    1. Temporal tiling: Split video into chunks with overlap (default: 81 frames, 10-frame overlap)
    2. Spatial tiling: Split each chunk into tiles with overlap (default: 832×480, 20-pixel overlap)
    3. WanVideo VACE processing: Process each tile through the complete pipeline
    4. Dimension-wise stitching:
       a) Column-wise: Stitch tiles vertically within each temporal chunk
       b) Line-wise: Stitch columns horizontally to complete temporal chunks  
       c) Temporal: Stitch temporal chunks together across time
    5. Cropping: Final video cropped to exact input dimensions
    6. Memory management: Proper model offloading between tiles to prevent leaks
    
    This enables processing of arbitrarily large videos that would otherwise exceed memory limits,
    while maintaining WanVideo's optimal performance and eliminating overlap artifacts through
    sophisticated dimension-wise fade blending.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Complete inputs combining tiling parameters with WanVideo VACE pipeline
        """
        return {
            "required": {
                # Input video and mask
                "video": ("IMAGE", {"tooltip": "Input video frames to process. Can be very large videos that will be automatically tiled for memory efficiency."}),
                "mask": ("MASK", {"tooltip": "Mask defining areas to be processed. Should match the video dimensions and frame count."}),
                
                # Core WanVideo pipeline inputs
                "model": ("WANVIDEOMODEL", {"tooltip": "WanVideo model for video generation/processing."}),
                "vae": ("WANVAE", {"tooltip": "Video VAE for encoding/decoding frames."}),
                
                # Tiling parameters
                "target_frames": ("INT", {"default": 81, "min": 16, "max": 200, "step": 1, "tooltip": "Target number of frames per temporal chunk. The model works best with 81 frames. If different, might generate artefact or frame drops"}),
                "target_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Target width for spatial tiles. Works best around 832."}),
                "target_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8, "tooltip": "Target height for spatial tiles. Works best around 480."}),
                "frame_overlap": ("INT", {"default": 10, "min": 0, "max": 40, "step": 1, "tooltip": "Number of overlapping frames between temporal chunks."}),
                "spatial_overlap": ("INT", {"default": 20, "min": 0, "max": 100, "step": 4, "tooltip": "Pixel overlap between spatial tiles."}),
                
                # WanVideoSampler core parameters
                "steps": ("INT", {"default": 30, "min": 1, "tooltip": "Number of diffusion sampling steps."}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "Classifier-free guidance strength. Higher values follow the conditioning more closely."}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": "Shift parameter for the sampling schedule."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducible results. Use the same seed to get identical outputs."}),
                "scheduler": (["unipc", "unipc/beta", "dpm++", "dpm++/beta","dpm++_sde", "dpm++_sde/beta", "euler", "euler/beta", "euler/accvideo", "deis", "lcm", "lcm/beta", "flowmatch_causvid", "flowmatch_distill", "multitalk"],
                    {"default": 'unipc', "tooltip": "Sampling scheduler algorithm."}),
                
                # WanVideoVACEEncode parameters  
                "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "VACE encoding strength. Controls how much the video is modified during processing. 1.0 mean total reconstruction, 0.0 means no change."}),
                "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # WanVideoDecode parameters
                "decode_enable_vae_tiling": ("BOOLEAN", {"default": False}),
                "decode_tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8}),
                "decode_tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8}),
                "decode_tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8}),
                "decode_tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8}),
                
                # Processing parameters
                "debug_mode": ("BOOLEAN", {"default": True}),
                "debug_color_shift": ("BOOLEAN", {"default": True}),
                "force_offload_between_tiles": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # WanVideoSampler optional inputs
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "samples": ("LATENT",),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "batched_cfg": ("BOOLEAN", {"default": False}),
                "rope_function": (["default", "comfy"], {"default": "comfy"}),
                
                # WanVideoSampler advanced optional inputs
                "feta_args": ("FETAARGS",),
                "context_options": ("CONTEXTOPTIONS",),
                "loop_args": ("LOOPARGS",),
                "sigmas": ("SIGMAS",),
                "unianimate_poses": ("UNIANIMATE_POSES",),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS",),
                "uni3c_embeds": ("UNI3C_EMBEDS",),
                "multitalk_embeds": ("MULTITALK_EMBEDS",),
                "freeinit_args": ("FREEINIT_ARGS",),
                
                # External pre-built arguments (for modularity)
                "cache_args": ("CACHEARGS",),
                "slg_args": ("SLGARGS",),
                "experimental_args": ("EXPERIMENTALARGS",),
                
                # WanVideoVACEEncode optional inputs  
                "vace_ref_images": ("IMAGE",),
                "vace_tiled_vae": ("BOOLEAN", {"default": False}),
                
                # WanVideoDecode optional inputs
                "decode_normalization": (["default", "minmax"], {"default": "default"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_video", "processing_info")
    OUTPUT_TOOLTIPS = ("Processed video with WanVideo VACE pipeline applied through tiled processing. Maintains original dimensions and quality.", 
                      "Detailed information about the processing including tile counts, timing, and processing summary.")
    FUNCTION = "process_tiled_wanvideo"
    OUTPUT_NODE = True
    CATEGORY = "TiledWan"
    DESCRIPTION = """
    Advanced video processing node that applies WanVideo VACE pipeline to large videos through intelligent tiling.
    
    Enables processing of arbitrarily large videos that would otherwise exceed memory limits.
    Uses dimension-wise tiling with temporal and spatial overlap to maintain quality and consistency.
    The complete WanVideo VACE pipeline is applied to each tile with sophisticated overwriting strategies
    for seamless results.

    Processing Algorithm:
    1. Temporal tiling: Split video into chunks with overlap (default: 81 frames, 10-frame overlap)
    2. Spatial tiling: Split each chunk into tiles with overlap (default: 832×480, 20-pixel overlap)  
    3. Temporal consistency: Previous chunks provide reference frames for upcoming chunks
    4. Spatial consistency: Already-processed neighboring tiles overwrite overlapping regions
    5. WanVideo VACE processing: Each tile processed through complete pipeline
    6. Dimension-wise stitching: Column-wise → Line-wise → Temporal stitching
    7. Final cropping: Output matches exact input dimensions

    Key Features:
    - Handles any video size through intelligent tiling
    - Temporal consistency across chunks via frame reference chaining
    - Spatial consistency through neighbor tile overwriting
    - Memory-efficient with model offloading between tiles
    - Complete WanVideo VACE pipeline integration
    - Sophisticated fade blending for seamless stitching
    - Tensor safety with hard copies to prevent data contamination
    
    Consistency Mechanisms:
    - Temporal: Last frames from previous chunks overwrite first frames of current chunks
    - Spatial: Left/top neighbors overwrite overlapping edges in current tiles
    - All overwritten regions have masks zeroed for proper processing
    """

    def process_tiled_wanvideo(self, video, mask, model, vae, target_frames, target_width, target_height, 
                              frame_overlap, spatial_overlap, steps, cfg, shift, seed, scheduler,
                              vace_strength, vace_start_percent, vace_end_percent, 
                              decode_enable_vae_tiling, decode_tile_x, decode_tile_y,
                              decode_tile_stride_x, decode_tile_stride_y, debug_mode, debug_color_shift,
                              force_offload_between_tiles, **kwargs):
        """
        Process large video through tiled WanVideo VACE pipeline with dimension-wise stitching
        """
        
        print("\n" + "="*80)
        print("              TILED WANVIDEO VACE PIPELINE (DIMENSION-WISE)")
        print("="*80)
        print("🚀 Starting dimension-wise tiled WanVideo VACE processing...")
        
        try:
            # Import WanVideo nodes
            custom_nodes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ComfyUI-WanVideoWrapper")
            if custom_nodes_path not in sys.path:
                sys.path.append(custom_nodes_path)
            
            parent_path = os.path.dirname(custom_nodes_path)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
            
            package_name = os.path.basename(custom_nodes_path)
            wanvideo_package = importlib.import_module(f"{package_name}.nodes")
            
            WanVideoSampler = wanvideo_package.WanVideoSampler
            WanVideoVACEEncode = wanvideo_package.WanVideoVACEEncode
            WanVideoDecode = wanvideo_package.WanVideoDecode
            print("✅ WanVideo nodes imported successfully!")
            
            # Input validation and preprocessing
            batch_size, height, width, channels = video.shape
            mask_batch, mask_height, mask_width = mask.shape
            
            print(f"📹 Input video shape: {video.shape} (B×H×W×C)")
            print(f"🎭 Input mask shape: {mask.shape} (B×H×W)")
            print(f"🎯 Target tile size: {target_frames} frames × {target_width}×{target_height}")
            print(f"🔗 Overlaps: {frame_overlap} frames, {spatial_overlap} pixels")
            
            # Validate mask dimensions
            if mask_batch != batch_size or mask_height != height or mask_width != width:
                print(f"⚠️  Warning: Mask dimensions {mask.shape} don't match video {video.shape}")
                print("   Resizing mask to match video...")
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1), size=(height, width), mode='nearest'
                ).squeeze(1)
                print(f"✅ Mask resized to: {mask.shape}")
            
            # STEP 1: Calculate tile dimensions using improved algorithm
            print(f"\n📏 STEP 1: Calculating optimal tile layout...")
            temporal_tiles = self._calculate_temporal_tiles(batch_size, target_frames, frame_overlap)
            spatial_tiles_h = self._calculate_spatial_tiles(height, target_height, spatial_overlap) 
            spatial_tiles_w = self._calculate_spatial_tiles(width, target_width, spatial_overlap)
            
            total_tiles = len(temporal_tiles) * len(spatial_tiles_h) * len(spatial_tiles_w)
            print(f"⏱️  Temporal chunks: {len(temporal_tiles)}")
            print(f"🗺️  Spatial tiles per frame: {len(spatial_tiles_h)}×{len(spatial_tiles_w)}")
            print(f"📦 Total tiles to process: {total_tiles}")
            
            # STEP 2: Extract all tiles and process through WanVideo VACE pipeline
            print(f"\n🧩 STEP 2: Extracting and processing {total_tiles} tiles through WanVideo VACE...")
            all_tiles = self._extract_and_process_wanvideo_tiles(
                video, mask, temporal_tiles, spatial_tiles_h, spatial_tiles_w,
                model, vae, WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                steps, cfg, shift, seed, scheduler, vace_strength, vace_start_percent, vace_end_percent,
                decode_enable_vae_tiling, decode_tile_x, decode_tile_y, decode_tile_stride_x, decode_tile_stride_y,
                debug_color_shift, force_offload_between_tiles, debug_mode, frame_overlap, spatial_overlap, kwargs
            )
            
            # STEP 3: Dimension-wise stitching - Column-wise (vertical stitching)
            print(f"\n🔄 STEP 3: Column-wise stitching (vertical)...")
            column_strips = self._stitch_columns(all_tiles, spatial_overlap, debug_mode)
            
            # STEP 4: Dimension-wise stitching - Line-wise (horizontal stitching)  
            print(f"\n🔄 STEP 4: Line-wise stitching (horizontal)...")
            temporal_chunks = self._stitch_lines(column_strips, spatial_overlap, debug_mode)
            
            # STEP 5: Dimension-wise stitching - Temporal stitching
            print(f"\n🔄 STEP 5: Temporal stitching...")
            stitched_video = self._stitch_temporal_chunks_new(temporal_chunks, temporal_tiles, frame_overlap)
            
            # STEP 6: Crop to original dimensions (ensure exact input size)
            print(f"\n✂️ STEP 6: Cropping to original dimensions...")
            print(f"📐 Stitched video shape: {stitched_video.shape}")
            print(f"🎯 Target shape: {video.shape}")
            final_video = stitched_video[:batch_size, :height, :width, :channels]
            print(f"✂️ Cropped to: {final_video.shape}")
            
            # Generate comprehensive processing info
            processing_summary = self._generate_wanvideo_processing_summary(
                all_tiles, temporal_tiles, spatial_tiles_h, spatial_tiles_w, total_tiles
            )
            
            print(f"✅ Dimension-wise tiled WanVideo VACE processing completed!")
            print(f"📤 Final video shape: {final_video.shape}")
            print(f"🧩 Total tiles processed: {len(all_tiles)}")
            successful_tiles = sum(1 for tile in all_tiles if hasattr(tile, 'processing_status') and tile.processing_status == 'success')
            print(f"✅ Successful tiles: {successful_tiles}/{total_tiles}")
            print("="*80 + "\n")
            
            # CLEANUP: Clear variables to prevent interference with subsequent runs
            try:
                del all_tiles, column_strips, temporal_chunks, stitched_video
                del temporal_tiles, spatial_tiles_h, spatial_tiles_w
                if 'previous_chunk_stitched_frame' in locals():
                    del previous_chunk_stitched_frame
                print("🧹 Variables cleaned up for next run")
            except:
                pass
            
            return (final_video, processing_summary)
            
        except Exception as e:
            print(f"❌ Error in tiled WanVideo VACE pipeline: {str(e)}")
            print(f"📋 Full traceback:")
            import traceback
            print(traceback.format_exc())
            print("="*80 + "\n")
            
            # Return original video in case of error
            error_info = f"Error during tiled WanVideo processing: {str(e)}"
            
            # CLEANUP: Clear variables even on error to prevent interference with subsequent runs
            try:
                if 'all_tiles' in locals():
                    del all_tiles
                if 'column_strips' in locals():
                    del column_strips
                if 'temporal_chunks' in locals():
                    del temporal_chunks
                if 'temporal_tiles' in locals():
                    del temporal_tiles
                if 'spatial_tiles_h' in locals():
                    del spatial_tiles_h
                if 'spatial_tiles_w' in locals():
                    del spatial_tiles_w
                if 'previous_chunk_stitched_frame' in locals():
                    del previous_chunk_stitched_frame
                print("🧹 Variables cleaned up after error")
            except:
                pass
            
            return (video, error_info)
    
    
    def _extract_and_process_wanvideo_tiles(self, video, mask, temporal_tiles, spatial_tiles_h, spatial_tiles_w,
                                        model, vae, WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                                        steps, cfg, shift, seed, scheduler, vace_strength, vace_start_percent, vace_end_percent,
                                        decode_enable_vae_tiling, decode_tile_x, decode_tile_y, decode_tile_stride_x, decode_tile_stride_y,
                                        debug_color_shift, force_offload_between_tiles, debug_mode, frame_overlap, spatial_overlap, kwargs):
        """
        Extract all tiles and process through WanVideo VACE pipeline with frame-wise temporal consistency
        """
        all_tiles = []
        
        # Store previous temporal chunk's COMPLETE STITCHED FRAME for reference
        # IMPORTANT: Always start fresh to avoid interference from previous runs
        previous_chunk_stitched_frame = None
        
        for temporal_idx, (t_start, t_end) in enumerate(temporal_tiles):
            if debug_mode:
                print(f"   🎬 Processing temporal chunk {temporal_idx + 1}/{len(temporal_tiles)} (frames {t_start}-{t_end-1})")
            
            # Extract temporal chunk
            video_chunk = video[t_start:t_end].clone()  # HARD COPY to avoid modifying original
            mask_chunk = mask[t_start:t_end].clone()    # HARD COPY to avoid modifying original
            
            # Store current chunk tiles for stitching the reference frame
            current_chunk_tiles = []
            
            # Process each spatial tile within this temporal chunk
            for h_idx, (h_start, h_end) in enumerate(spatial_tiles_h):
                for w_idx, (w_start, w_end) in enumerate(spatial_tiles_w):
                    tile_idx = len(all_tiles)  # Global tile index
                    
                    if debug_mode:
                        print(f"      🧩 Tile {tile_idx + 1}: T{temporal_idx}:L{h_idx}:C{w_idx} "
                            f"H[{h_start}:{h_end}] × W[{w_start}:{w_end}]")
                    
                    # Extract spatial tiles from temporal chunk
                    video_tile = video_chunk[:, h_start:h_end, w_start:w_end, :].clone()  # HARD COPY to avoid modifying chunk
                    mask_tile = mask_chunk[:, h_start:h_end, w_start:w_end].clone()        # HARD COPY to avoid modifying chunk
                    
                    # ========== TEMPORAL OVERWRITING FOR CONSISTENCY ==========
                    if temporal_idx > 0 and previous_chunk_stitched_frame is not None:
                        # Determine how many frames to overwrite
                        frames_to_overwrite = frame_overlap
                        if temporal_idx == len(temporal_tiles) - 1:
                            # Last chunk:
                            # Look at previous chunk end frame
                            previous_chunk_end_frame_index = temporal_tiles[temporal_idx - 1][1] - 1
                            # Use as many frame as actual overlap allows. 
                            frames_to_overwrite = previous_chunk_end_frame_index - t_start + 1
                        
                        # Extract corresponding frames from previous chunk's stitched result
                        prev_chunk_frames = previous_chunk_stitched_frame.shape[0]
                        source_start_idx = max(0, prev_chunk_frames - frames_to_overwrite)
                        source_frames = previous_chunk_stitched_frame[source_start_idx:, h_start:h_end, w_start:w_end, :].clone()  # HARD COPY
                        
                        # Overwrite first frames of current tile with last frames from previous chunk
                        if frames_to_overwrite > 0:
                            video_tile[:frames_to_overwrite] = source_frames
                            # Zero out corresponding mask areas
                            mask_tile[:frames_to_overwrite] = 0.0
                            
                            if debug_mode:
                                print(f"         🔄 Temporal overwrite: {frames_to_overwrite} frames from prev chunk")

                    # ========== SPATIAL OVERWRITING FOR CONSISTENCY ==========
                    # Look for already processed tiles to overwrite overlapping regions
                    
                    # Check for LEFT neighbor (same temporal and line, previous column)
                    if w_idx > 0:
                        left_neighbor = None
                        for tile in current_chunk_tiles:
                            if (tile.temporal_index == temporal_idx and 
                                tile.line_index == h_idx and 
                                tile.column_index == w_idx - 1):
                                left_neighbor = tile
                                break
                        
                        if left_neighbor is not None and hasattr(left_neighbor, 'content'):
                            # Calculate overlap region
                            left_tile_end = left_neighbor.spatial_range_w[1]
                            current_tile_start = w_start
                            overlap_width = left_tile_end - current_tile_start
                            
                            if overlap_width > 0:
                                # Extract overlap from left neighbor's right edge
                                left_content = left_neighbor.content
                                left_overlap = left_content[:, :, -overlap_width:, :]  # Last overlap_width columns
                                
                                # Overwrite left edge of current tile
                                video_tile[:, :, :overlap_width, :] = left_overlap
                                # Zero out corresponding mask areas
                                mask_tile[:, :, :overlap_width] = 0.0
                                
                                if debug_mode:
                                    print(f"         ↔️ Spatial overwrite LEFT: {overlap_width} pixels from neighbor")
                    
                    # Check for TOP neighbor (same temporal and column, previous line)
                    if h_idx > 0:
                        top_neighbor = None
                        for tile in current_chunk_tiles:
                            if (tile.temporal_index == temporal_idx and 
                                tile.line_index == h_idx - 1 and 
                                tile.column_index == w_idx):
                                top_neighbor = tile
                                break
                        
                        if top_neighbor is not None and hasattr(top_neighbor, 'content'):
                            # Calculate overlap region
                            top_tile_end = top_neighbor.spatial_range_h[1]
                            current_tile_start = h_start
                            overlap_height = top_tile_end - current_tile_start
                            
                            if overlap_height > 0:
                                # Extract overlap from top neighbor's bottom edge
                                top_content = top_neighbor.content
                                top_overlap = top_content[:, -overlap_height:, :, :]  # Last overlap_height rows
                                
                                # Overwrite top edge of current tile
                                video_tile[:, :overlap_height, :, :] = top_overlap
                                # Zero out corresponding mask areas
                                mask_tile[:, :overlap_height, :] = 0.0
                                
                                if debug_mode:
                                    print(f"         ↕️ Spatial overwrite TOP: {overlap_height} pixels from neighbor")
                    
                    # Handle TOP-LEFT corner conflict (if both top and left neighbors exist)
                    if h_idx > 0 and w_idx > 0:
                        # Find top-left diagonal neighbor
                        topleft_neighbor = None
                        for tile in current_chunk_tiles:
                            if (tile.temporal_index == temporal_idx and 
                                tile.line_index == h_idx - 1 and 
                                tile.column_index == w_idx - 1):
                                topleft_neighbor = tile
                                break
                        
                        if topleft_neighbor is not None and hasattr(topleft_neighbor, 'content'):
                            # Calculate corner overlap
                            top_tile_end_h = topleft_neighbor.spatial_range_h[1]
                            left_tile_end_w = topleft_neighbor.spatial_range_w[1]
                            current_start_h = h_start
                            current_start_w = w_start
                            
                            overlap_h = top_tile_end_h - current_start_h
                            overlap_w = left_tile_end_w - current_start_w
                            
                            if overlap_h > 0 and overlap_w > 0:
                                # Extract corner from top-left neighbor
                                topleft_content = topleft_neighbor.content
                                corner_overlap = topleft_content[:, -overlap_h:, -overlap_w:, :]
                                
                                # Overwrite corner of current tile
                                video_tile[:, :overlap_h, :overlap_w, :] = corner_overlap
                                # Zero out corresponding mask areas
                                mask_tile[:, :overlap_h, :overlap_w] = 0.0
                                
                                if debug_mode:
                                    print(f"         ↖️ Spatial overwrite CORNER: {overlap_h}x{overlap_w} from diagonal neighbor")
                    
                    # FRAME-WISE TEMPORAL CONSISTENCY: Determine reference image for this tile
                    tile_ref_images = None
                    
                    if temporal_idx == 0:
                        # First temporal chunk: Use user-provided ref_images (if any)
                        tile_ref_images = kwargs.get("vace_ref_images")
                        if debug_mode and h_idx == 0 and w_idx == 0:  # Only print once per chunk
                            print(f"      🎯 First chunk: Using user-provided ref_images")
                    else:
                        # Subsequent chunks: Use COMPLETE STITCHED FRAME from previous chunk as reference
                        if previous_chunk_stitched_frame is not None:
                            # Extract reference frame: frame_overlap frames before the last frame
                            ref_frame_idx = previous_chunk_stitched_frame.shape[0] - frame_overlap - 1
                            ref_frame_idx = max(0, ref_frame_idx)  # Ensure non-negative
                            
                            # Extract single COMPLETE FRAME as reference image
                            tile_ref_images = previous_chunk_stitched_frame[ref_frame_idx:ref_frame_idx+1].clone()  # Shape: [1, H, W, C] - HARD COPY
                            
                            if debug_mode and h_idx == 0 and w_idx == 0:  # Only print once per chunk
                                print(f"      🔗 Frame-wise temporal chain: Using complete stitched frame {ref_frame_idx} from previous chunk T{temporal_idx-1}")
                                print(f"         Reference frame shape: {tile_ref_images.shape}")
                        else:
                            # Fallback: Use user-provided ref_images
                            tile_ref_images = kwargs.get("vace_ref_images")
                            if debug_mode and h_idx == 0 and w_idx == 0:
                                print(f"      ⚠️  No previous stitched frame found, using user ref_images as fallback")
                    
                    # Process tile through WanVideo VACE pipeline with frame-wise temporal reference
                    try:
                        # Create kwargs with tile-specific reference image (COMPLETE FRAME)
                        tile_kwargs = kwargs.copy()
                        tile_kwargs["vace_ref_images"] = tile_ref_images
                        
                        processed_tile, tile_latents = self._process_tile_through_wanvideo(
                            video_tile, mask_tile, model, vae, 
                            WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                            steps, cfg, shift, seed + tile_idx,  # Unique seed per tile
                            scheduler, vace_strength, vace_start_percent, vace_end_percent,
                            decode_enable_vae_tiling, decode_tile_x, decode_tile_y,
                            decode_tile_stride_x, decode_tile_stride_y, debug_color_shift, tile_kwargs
                        )
                        
                        # Create Tile object with processed content
                        tile = Tile(
                            content=processed_tile,
                            temporal_index=temporal_idx,
                            line_index=h_idx,
                            column_index=w_idx,
                            temporal_range=(t_start, t_end),
                            spatial_range_h=(h_start, h_end),
                            spatial_range_w=(w_start, w_end)
                        )
                        
                        # Store processing status and reference info for tracking
                        tile.processing_status = 'success'
                        tile.seed_used = seed + tile_idx
                        tile.latents = tile_latents
                        tile.ref_frame_source = f"complete_frame_T{temporal_idx-1}" if temporal_idx > 0 and previous_chunk_stitched_frame is not None else "user_provided"
                        
                        all_tiles.append(tile)
                        current_chunk_tiles.append(tile)
                        
                        # Force memory cleanup between tiles if requested
                        if force_offload_between_tiles:
                            self._force_memory_cleanup(model, vae)
                        
                        # CLEANUP: Delete tile copies to free memory immediately
                        try:
                            del video_tile, mask_tile, processed_tile
                            if 'tile_latents' in locals():
                                del tile_latents
                        except:
                            pass
                            
                    except Exception as tile_error:
                        print(f"         ❌ Error processing tile {tile_idx + 1}: {str(tile_error)}")
                        
                        # Create fallback Tile object with original content
                        tile = Tile(
                            content=video_tile,  # Use original video tile as fallback
                            temporal_index=temporal_idx,
                            line_index=h_idx,
                            column_index=w_idx,
                            temporal_range=(t_start, t_end),
                            spatial_range_h=(h_start, h_end),
                            spatial_range_w=(w_start, w_end)
                        )
                        
                        # Store error status for tracking
                        tile.processing_status = f'failed: {str(tile_error)}'
                        tile.seed_used = seed + tile_idx
                        tile.latents = None
                        tile.ref_frame_source = "error_fallback"
                        
                        all_tiles.append(tile)
                        current_chunk_tiles.append(tile)
                        
                        # CLEANUP: Delete tile copies to free memory immediately (error case)
                        try:
                            del video_tile, mask_tile
                        except:
                            pass
            
            # STITCH THE CURRENT CHUNK TO CREATE REFERENCE FOR NEXT CHUNK
            if debug_mode:
                print(f"      🔧 Stitching current chunk T{temporal_idx} for next chunk's reference...")
            
            # Group current chunk tiles for stitching
            current_chunk_tiles_for_stitching = [tile for tile in current_chunk_tiles]
            
            # Perform dimension-wise stitching for this temporal chunk
            try:
                # Step 1: Column-wise stitching
                column_strips = self._stitch_columns(current_chunk_tiles_for_stitching, spatial_overlap, False)  # No debug for reference stitching
                
                # Step 2: Line-wise stitching  
                temporal_chunks = self._stitch_lines(column_strips, spatial_overlap, False)  # No debug for reference stitching
                
                # Get the stitched chunk content (should be only one chunk)
                if temporal_chunks and len(temporal_chunks) > 0:
                    current_chunk_stitched = temporal_chunks[0]['content']
                    
                    # Store this stitched chunk as reference for next temporal chunk
                    previous_chunk_stitched_frame = current_chunk_stitched
                    
                    if debug_mode:
                        print(f"      ✅ Chunk T{temporal_idx} stitched for reference: {current_chunk_stitched.shape}")
                else:
                    print(f"      ⚠️  Warning: Failed to stitch chunk T{temporal_idx} for reference")
                    previous_chunk_stitched_frame = None
                    
            except Exception as stitching_error:
                print(f"      ❌ Error stitching chunk T{temporal_idx} for reference: {str(stitching_error)}")
                previous_chunk_stitched_frame = None
            
            if debug_mode:
                chain_status = "user_ref" if temporal_idx == 0 else f"complete_frame_from_T{temporal_idx-1}"
                print(f"      ✅ Temporal chunk {temporal_idx} completed with {chain_status} references")
            
            # CLEANUP: Delete temporal chunk copies to free memory
            try:
                del video_chunk, mask_chunk
                del current_chunk_tiles, current_chunk_tiles_for_stitching
                if 'column_strips' in locals():
                    del column_strips
                if 'temporal_chunks' in locals():
                    del temporal_chunks
                if 'current_chunk_stitched' in locals():
                    del current_chunk_stitched
            except:
                pass
        
        print(f"   ✅ Extracted and processed {len(all_tiles)} tiles through WanVideo VACE pipeline")
        successful_tiles = sum(1 for tile in all_tiles if tile.processing_status == 'success')
        print(f"   🎯 Success rate: {successful_tiles}/{len(all_tiles)} ({(successful_tiles/len(all_tiles))*100:.1f}%)")
        
        # Debug: Print overwriting statistics
        if debug_mode:
            temporal_overwrites = sum(1 for t_idx in range(1, len(temporal_tiles)))
            spatial_overwrites = 0
            corner_overwrites = 0
            
            for temporal_idx in range(len(temporal_tiles)):
                for h_idx in range(len(spatial_tiles_h)):
                    for w_idx in range(len(spatial_tiles_w)):
                        if w_idx > 0:  # Has left neighbor
                            spatial_overwrites += 1
                        if h_idx > 0:  # Has top neighbor
                            spatial_overwrites += 1
                        if h_idx > 0 and w_idx > 0:  # Has corner conflict
                            corner_overwrites += 1
            
            print(f"   🔄 Overwriting consistency enhancements:")
            print(f"      • Temporal overwrites: {temporal_overwrites * len(spatial_tiles_h) * len(spatial_tiles_w)} tile operations")
            print(f"      • Spatial overwrites: {spatial_overwrites} edge operations") 
            print(f"      • Corner overwrites: {corner_overwrites} corner conflict resolutions")
        
        # Debug: Print frame-wise temporal consistency chain summary
        if debug_mode:
            print(f"   🖼️  Frame-wise temporal consistency chain summary:")
            print(f"      • Chunk 0: Used user-provided reference images")
            for t_idx in range(1, len(temporal_tiles)):
                print(f"      • Chunk {t_idx}: Used complete stitched frame from Chunk {t_idx-1}")
        
        # CLEANUP: Ensure previous_chunk_stitched_frame doesn't persist for next run
        try:
            del previous_chunk_stitched_frame
        except:
            pass
        
        return all_tiles
    
    def _process_tile_through_wanvideo(self, video_tile, mask_tile, model, vae,
                                    WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                                    steps, cfg, shift, seed, scheduler, vace_strength,
                                    vace_start_percent, vace_end_percent, decode_enable_vae_tiling,
                                    decode_tile_x, decode_tile_y, decode_tile_stride_x, 
                                    decode_tile_stride_y, debug_color_shift, kwargs):
        """Process a single tile through the complete WanVideo VACE pipeline with tile-specific reference"""
        
        tile_frames, tile_height, tile_width = video_tile.shape[:3]
        
        # Get tile-specific reference images (from temporal consistency chain)
        tile_ref_images = kwargs.get("vace_ref_images")
        
        # Step 1: Create VACE embeds for this tile with its specific reference
        vace_node = WanVideoVACEEncode()
        vace_embeds = vace_node.process(
            vae=vae,
            width=tile_width,
            height=tile_height, 
            num_frames=tile_frames,
            strength=vace_strength,
            vace_start_percent=vace_start_percent,
            vace_end_percent=vace_end_percent,
            input_frames=video_tile,        # Use tile as input frames
            ref_images=tile_ref_images,     # Use tile-specific reference (from chain or user)
            input_masks=mask_tile,          # Use tile mask
            tiled_vae=kwargs.get("vace_tiled_vae", False)
        )[0]
        
        # Step 2: Run WanVideoSampler on this tile
        sampler_node = WanVideoSampler()
        latent_samples = sampler_node.process(
            model=model,
            image_embeds=vace_embeds,
            steps=steps,
            cfg=cfg,
            shift=shift,
            seed=seed,
            scheduler=scheduler,
            riflex_freq_index=kwargs.get("riflex_freq_index", 0),
            text_embeds=kwargs.get("text_embeds"),
            samples=kwargs.get("samples"),
            denoise_strength=kwargs.get("denoise_strength", 1.0),
            force_offload=kwargs.get("force_offload", True),
            
            # External arguments
            cache_args=kwargs.get("cache_args"),
            slg_args=kwargs.get("slg_args"),
            experimental_args=kwargs.get("experimental_args"),
            
            # Other arguments
            feta_args=kwargs.get("feta_args"),
            context_options=kwargs.get("context_options"),
            flowedit_args=None,
            batched_cfg=kwargs.get("batched_cfg", False),
            rope_function=kwargs.get("rope_function", "comfy"),
            loop_args=kwargs.get("loop_args"),
            sigmas=kwargs.get("sigmas"),
            unianimate_poses=kwargs.get("unianimate_poses"),
            fantasytalking_embeds=kwargs.get("fantasytalking_embeds"),
            uni3c_embeds=kwargs.get("uni3c_embeds"),
            multitalk_embeds=kwargs.get("multitalk_embeds"),
            freeinit_args=kwargs.get("freeinit_args")
        )[0]
        
        # Step 3: Decode latents back to video for this tile
        decode_node = WanVideoDecode()
        processed_tile = decode_node.decode(
            vae=vae,
            samples=latent_samples,
            enable_vae_tiling=decode_enable_vae_tiling,
            tile_x=decode_tile_x,
            tile_y=decode_tile_y,
            tile_stride_x=decode_tile_stride_x,
            tile_stride_y=decode_tile_stride_y,
            normalization=kwargs.get("decode_normalization", "default")
        )[0]
        
        # Step 4: Apply debug color shift to identify tiles visually (if enabled)
        if debug_color_shift:
            debug_color_shifted_tile = self._apply_debug_color_shift(processed_tile, seed)
            return debug_color_shifted_tile, latent_samples
        else:
            return processed_tile, latent_samples
    
    def _apply_debug_color_shift(self, tile, seed):
        """Apply a random color shift to each tile for debugging visualization"""
        import random
        
        # Generate truly random RGB shifts (different each time, not based on seed)
        r_shift = (random.random() - 0.5) * 0.3  # ±0.15 range
        g_shift = (random.random() - 0.5) * 0.3  # ±0.15 range  
        b_shift = (random.random() - 0.5) * 0.3  # ±0.15 range
        
        print(f"      🎨 Debug color shift - Tile: R{r_shift:+.3f}, G{g_shift:+.3f}, B{b_shift:+.3f}")
        
        # Apply color shift to the tile
        shifted_tile = tile.clone()
        
        if tile.shape[-1] >= 3:  # Ensure we have RGB channels
            shifted_tile[:, :, :, 0] = torch.clamp(shifted_tile[:, :, :, 0] + r_shift, 0, 1)
            shifted_tile[:, :, :, 1] = torch.clamp(shifted_tile[:, :, :, 1] + g_shift, 0, 1)
            shifted_tile[:, :, :, 2] = torch.clamp(shifted_tile[:, :, :, 2] + b_shift, 0, 1)
        
        return shifted_tile

    def _stitch_columns(self, all_tiles, spatial_overlap, debug_mode):
        """
        Stitch tiles vertically (same column, same temporal chunk) to create column strips
        """
        # Group tiles by temporal_index and column_index
        column_groups = {}
        for tile in all_tiles:
            key = (tile.temporal_index, tile.column_index)
            if key not in column_groups:
                column_groups[key] = []
            column_groups[key].append(tile)
        
        # Sort each group by line_index
        for key in column_groups:
            column_groups[key].sort(key=lambda t: t.line_index)
        
        column_strips = []
        
        for (temporal_idx, col_idx), column_tiles in column_groups.items():
            if debug_mode and temporal_idx == 0:
                print(f"      🔄 Column {col_idx}: {len(column_tiles)} tiles")
            
            # Stitch tiles in this column vertically
            if len(column_tiles) == 1:
                column_strip = column_tiles[0].content
            else:
                column_strip = self._stitch_tiles_vertically(column_tiles, spatial_overlap)
            
            # Store column strip with its position info
            strip_info = {
                'content': column_strip,
                'temporal_index': temporal_idx,
                'column_index': col_idx,
                'spatial_range_w': column_tiles[0].spatial_range_w,
                'spatial_range_h': (column_tiles[0].spatial_range_h[0], column_tiles[-1].spatial_range_h[1])
            }
            column_strips.append(strip_info)
        
        print(f"   ✅ Created {len(column_strips)} column strips")
        return column_strips
    

    def _stitch_lines(self, column_strips, spatial_overlap, debug_mode):
        """
        Stitch column strips horizontally (same temporal chunk) to create complete temporal chunks
        """
        # Group column strips by temporal_index
        temporal_groups = {}
        for strip in column_strips:
            temporal_idx = strip['temporal_index']
            if temporal_idx not in temporal_groups:
                temporal_groups[temporal_idx] = []
            temporal_groups[temporal_idx].append(strip)
        
        # Sort each group by column_index
        for temporal_idx in temporal_groups:
            temporal_groups[temporal_idx].sort(key=lambda s: s['column_index'])
        
        temporal_chunks = []
        
        for temporal_idx, strips in temporal_groups.items():
            if debug_mode and temporal_idx == 0:
                print(f"      🔄 Temporal chunk {temporal_idx}: {len(strips)} strips")
            
            # Stitch column strips horizontally
            if len(strips) == 1:
                chunk_content = strips[0]['content']
            else:
                chunk_content = self._stitch_strips_horizontally(strips, spatial_overlap)
            
            # Store temporal chunk with its info
            chunk_info = {
                'content': chunk_content,
                'temporal_index': temporal_idx,
                'spatial_range_h': strips[0]['spatial_range_h'],
                'spatial_range_w': (strips[0]['spatial_range_w'][0], strips[-1]['spatial_range_w'][1])
            }
            temporal_chunks.append(chunk_info)
        
        # Sort by temporal index
        temporal_chunks.sort(key=lambda c: c['temporal_index'])
        
        print(f"   ✅ Created {len(temporal_chunks)} temporal chunks")
        return temporal_chunks

    def _stitch_tiles_vertically(self, column_tiles, spatial_overlap):
        """
        Stitch tiles vertically (in height dimension) with proper fade blending.
        Always respects spatial_overlap limit - starts blending from the correct position
        within the overlapping region when actual overlap exceeds spatial_overlap.
        """
        if len(column_tiles) == 1:
            return column_tiles[0].content
        
        # Get dimensions
        first_tile = column_tiles[0].content
        total_height = sum(tile.spatial_range_h[1] - tile.spatial_range_h[0] for tile in column_tiles)
        # Subtract overlaps
        total_height -= spatial_overlap * (len(column_tiles) - 1)
        
        width = first_tile.shape[2]
        channels = first_tile.shape[3]
        frames = first_tile.shape[0]
        
        # Create result tensor
        result = torch.zeros((frames, total_height, width, channels), 
                           dtype=first_tile.dtype, device=first_tile.device)
        
        current_h = 0
        for i, tile in enumerate(column_tiles):
            tile_h = tile.content.shape[1]
            
            if i == 0:
                # First tile - place entirely
                result[:, current_h:current_h+tile_h, :, :] = tile.content
                current_h += tile_h
            else:
                # Subsequent tiles - use full available overlap for blending
                expected_start = tile.spatial_range_h[0]
                actual_overlap = current_h - expected_start
                actual_overlap = max(0, actual_overlap)  # Ensure non-negative
                
                # Use full overlap for better blending quality
                overlap_h = min(actual_overlap, tile_h // 2, current_h)
                
                if overlap_h > 0:
                    # Get regions for blending (using full overlap)
                    result_overlap = result[:, current_h-overlap_h:current_h, :, :]
                    tile_overlap = tile.content[:, :overlap_h, :, :]
                    
                    # Create vertical fade mask
                    fade_mask = self._create_vertical_fade_mask(overlap_h, result_overlap.dtype, result_overlap.device)
                    
                    # Apply fade blending
                    blended_overlap = result_overlap * (1.0 - fade_mask) + tile_overlap * fade_mask
                    result[:, current_h-overlap_h:current_h, :, :] = blended_overlap
                    
                    # Place the non-overlapping part
                    if tile_h > overlap_h:
                        result[:, current_h:current_h+tile_h-overlap_h, :, :] = tile.content[:, overlap_h:, :, :]
                    current_h += tile_h - overlap_h
                else:
                    # No overlap - place directly
                    result[:, current_h:current_h+tile_h, :, :] = tile.content
                    current_h += tile_h
        
        return result
    
    def _stitch_strips_horizontally(self, strips, spatial_overlap):
        """
        Stitch column strips horizontally (in width dimension) with proper fade blending.
        Always respects spatial_overlap limit - starts blending from the correct position
        within the overlapping region when actual overlap exceeds spatial_overlap.
        """
        if len(strips) == 1:
            return strips[0]['content']
        
        # Get dimensions
        first_strip = strips[0]['content']
        total_width = sum(s['spatial_range_w'][1] - s['spatial_range_w'][0] for s in strips)
        # Subtract overlaps
        total_width -= spatial_overlap * (len(strips) - 1)
        
        height = first_strip.shape[1]
        channels = first_strip.shape[3]
        frames = first_strip.shape[0]
        
        # Create result tensor
        result = torch.zeros((frames, height, total_width, channels), 
                           dtype=first_strip.dtype, device=first_strip.device)
        
        current_w = 0
        for i, strip in enumerate(strips):
            strip_content = strip['content']
            strip_w = strip_content.shape[2]
            
            if i == 0:
                # First strip - place entirely
                result[:, :, current_w:current_w+strip_w, :] = strip_content
                current_w += strip_w
            else:
                # Subsequent strips - use full available overlap for blending
                expected_start = strip['spatial_range_w'][0]
                actual_overlap = current_w - expected_start
                actual_overlap = max(0, actual_overlap)  # Ensure non-negative
                
                # Use full overlap for better blending quality
                overlap_w = min(actual_overlap, strip_w // 2, current_w)
                
                if overlap_w > 0:
                    # Get regions for blending (using full overlap)
                    result_overlap = result[:, :, current_w-overlap_w:current_w, :]
                    strip_overlap = strip_content[:, :, :overlap_w, :]
                    
                    # Create horizontal fade mask
                    fade_mask = self._create_horizontal_fade_mask(overlap_w, result_overlap.dtype, result_overlap.device)
                    
                    # Apply fade blending
                    blended_overlap = result_overlap * (1.0 - fade_mask) + strip_overlap * fade_mask
                    result[:, :, current_w-overlap_w:current_w, :] = blended_overlap
                    
                    # Place the non-overlapping part
                    if strip_w > overlap_w:
                        result[:, :, current_w:current_w+strip_w-overlap_w, :] = strip_content[:, :, overlap_w:, :]
                    current_w += strip_w - overlap_w
                else:
                    # No overlap - place directly
                    result[:, :, current_w:current_w+strip_w, :] = strip_content
                    current_w += strip_w
        
        return result
    
    def _create_vertical_fade_mask(self, fade_size, dtype, device):
        """Create a vertical fade mask for blending tiles vertically."""
        fade_mask = torch.linspace(0, 1, fade_size, dtype=dtype, device=device)
        # Shape: (1, fade_size, 1, 1) for broadcasting
        return fade_mask.view(1, fade_size, 1, 1)
    
    def _create_horizontal_fade_mask(self, fade_size, dtype, device):
        """Create a horizontal fade mask for blending tiles horizontally."""
        fade_mask = torch.linspace(0, 1, fade_size, dtype=dtype, device=device)
        # Shape: (1, 1, fade_size, 1) for broadcasting
        return fade_mask.view(1, 1, fade_size, 1)
    
    def _stitch_temporal_chunks_new(self, temporal_chunks, temporal_tiles, frame_overlap):
        """
        Stitch temporal chunks to create the final output with temporal blending.
        Always respects frame_overlap limit - starts blending from the correct position
        within the overlapping region when actual overlap exceeds frame_overlap.
        """
        if not temporal_chunks:
            return None
            
        if len(temporal_chunks) == 1:
            return temporal_chunks[0]['content']
        
        # Get dimensions from first chunk
        first_chunk_content = temporal_chunks[0]['content']
        
        # Calculate total frames accounting for overlaps
        total_frames = sum(chunk['content'].shape[0] for chunk in temporal_chunks)
        if frame_overlap > 0:
            total_frames -= (len(temporal_chunks) - 1) * frame_overlap
        
        # Get spatial dimensions from the stitched chunks (not original input)
        height = first_chunk_content.shape[1]
        width = first_chunk_content.shape[2]
        channels = first_chunk_content.shape[3]
        
        print(f"   📏 Temporal stitching dimensions:")
        print(f"      • Total frames: {total_frames}")
        print(f"      • Spatial size: {height}×{width}")
        print(f"      • Channels: {channels}")
        print(f"      • Chunk sizes: {[chunk['content'].shape for chunk in temporal_chunks]}")
        
        # Initialize result tensor with correct dimensions
        result = torch.zeros(
            (total_frames, height, width, channels),
            dtype=first_chunk_content.dtype,
            device=first_chunk_content.device
        )
        
        current_t = 0
        for i, chunk_info in enumerate(temporal_chunks):
            chunk_content = chunk_info['content']
            chunk_frames = chunk_content.shape[0]
            
            if i == 0:
                # First chunk - place entirely
                result[current_t:current_t+chunk_frames] = chunk_content
                current_t += chunk_frames
            else:
                # Subsequent chunks - use full available overlap for blending
                temporal_tile_info = temporal_tiles[i]
                expected_start = temporal_tile_info[0]
                actual_overlap = current_t - expected_start
                actual_overlap = max(0, actual_overlap)  # Ensure non-negative
                
                # Use full overlap for better blending quality
                overlap_frames = min(actual_overlap, chunk_frames // 2, current_t)
                
                if overlap_frames > 0:
                    # Get regions for blending (using full overlap)
                    result_overlap = result[current_t-overlap_frames:current_t]
                    chunk_overlap = chunk_content[:overlap_frames]
                    
                    # Create temporal fade mask
                    fade_mask = self._create_temporal_fade_mask(overlap_frames, result_overlap.dtype, result_overlap.device)
                    
                    # Apply fade blending
                    blended_overlap = result_overlap * (1.0 - fade_mask) + chunk_overlap * fade_mask
                    result[current_t-overlap_frames:current_t] = blended_overlap
                    
                    # Place the non-overlapping part
                    if chunk_frames > overlap_frames:
                        result[current_t:current_t+chunk_frames-overlap_frames] = chunk_content[overlap_frames:]
                    current_t += chunk_frames - overlap_frames
                else:
                    # No overlap - place directly
                    result[current_t:current_t+chunk_frames] = chunk_content
                    current_t += chunk_frames
        
        return result
    
    def _create_temporal_fade_mask(self, fade_size, dtype, device):
        """Create a temporal fade mask for blending across time dimension."""
        fade_mask = torch.linspace(0, 1, fade_size, dtype=dtype, device=device)
        # Shape: (fade_size, 1, 1, 1) for broadcasting
        return fade_mask.view(fade_size, 1, 1, 1)
    
    def _force_memory_cleanup(self, model, vae):
        """Force memory cleanup between tile processing"""
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Try to offload models if they have offload methods
            if hasattr(model, 'cpu_offload'):
                model.cpu_offload()
            elif hasattr(model, 'to'):
                model.to('cpu')
                
            if hasattr(vae, 'cpu_offload'):
                vae.cpu_offload()
            elif hasattr(vae, 'to'):
                vae.to('cpu')
                
        except Exception as cleanup_error:
            print(f"      ⚠️  Warning: Memory cleanup failed: {str(cleanup_error)}")
    
    def _calculate_temporal_tiles(self, total_frames, target_frames, overlap):
        """Calculate temporal tile ranges with overlap handling"""
        tiles = []
        
        if total_frames <= target_frames:
            tiles.append((0, total_frames))
        else:
            stride = target_frames - overlap
            current = 0
            
            while current < total_frames:
                end = min(current + target_frames, total_frames)
                remaining = total_frames - end
                
                if remaining > 0:
                    tiles.append((current, end))
                    if remaining < stride:
                        final_start = total_frames - target_frames
                        tiles.append((final_start, total_frames))
                        break
                else:
                    tiles.append((current, end))
                    break
                    
                current += stride
        
        return tiles
    
    def _calculate_spatial_tiles(self, total_size, target_size, overlap):
        """Calculate spatial tile ranges with overlap handling"""
        tiles = []
        
        if total_size <= target_size:
            tiles.append((0, total_size))
        else:
            stride = target_size - overlap
            current = 0
            
            while current < total_size:
                end = min(current + target_size, total_size)
                remaining = total_size - end
                
                if remaining > 0:
                    tiles.append((current, end))
                    if remaining < stride:
                        final_start = total_size - target_size
                        tiles.append((final_start, total_size))
                        break
                else:
                    tiles.append((current, end))
                    break
                    
                current += stride
        
        return tiles
    
    def _place_tile_with_overlap(self, target, tile, h_start, h_end, w_start, w_end, spatial_overlap):
        """Place tile into target tensor, handling overlaps with spatial fade blending"""
        tile_h, tile_w = tile.shape[1:3]
        
        # Get the target region
        target_region = target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :]
        
        # Check if there's existing content (non-zero) in the target area
        if target_region.sum() > 0:
            # Use production-quality fade blending
            fade_mask = self._create_spatial_fade_mask(tile_h, tile_w, target_region, tile, spatial_overlap)
            
            # Apply fade blending: existing * (1 - fade_mask) + tile * fade_mask
            blended = target_region * (1.0 - fade_mask) + tile * fade_mask
            target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :] = blended
        else:
            # First tile in this area - place directly
            target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :] = tile
    
    def _create_spatial_fade_mask(self, tile_h, tile_w, existing, new_tile, spatial_overlap):
        """Create a spatial fade mask for smooth blending between tiles"""
        # Create base mask (1.0 = use new tile, 0.0 = use existing)
        mask = torch.ones(1, tile_h, tile_w, 1, dtype=existing.dtype, device=existing.device)
        
        # Define fade distance using spatial_overlap parameter
        fade_h = min(spatial_overlap, tile_h // 2)  # Fade over spatial_overlap pixels or half tile height
        fade_w = min(spatial_overlap, tile_w // 2)  # Fade over spatial_overlap pixels or half tile width
        
        # Check which edges have existing content
        has_top = existing[:, :fade_h, :, :].sum() > 0
        has_bottom = existing[:, -fade_h:, :, :].sum() > 0
        has_left = existing[:, :, :fade_w, :].sum() > 0
        has_right = existing[:, :, -fade_w:, :].sum() > 0
        
        # Create fade gradients for each edge
        if has_top:
            for i in range(fade_h):
                alpha = i / fade_h
                mask[:, i, :, :] = alpha
        
        if has_bottom:
            for i in range(fade_h):
                alpha = 1.0 - (i / fade_h)
                mask[:, tile_h - 1 - i, :, :] = alpha
        
        if has_left:
            for i in range(fade_w):
                alpha = i / fade_w
                mask[:, :, i, :] = torch.minimum(mask[:, :, i, :], torch.tensor(alpha, dtype=mask.dtype, device=mask.device))
        
        if has_right:
            for i in range(fade_w):
                alpha = 1.0 - (i / fade_w)
                mask[:, :, tile_w - 1 - i, :] = torch.minimum(mask[:, :, tile_w - 1 - i, :], torch.tensor(alpha, dtype=mask.dtype, device=mask.device))
        
        return mask.expand_as(new_tile)
    
    def _stitch_temporal_chunks(self, chunks, temporal_tiles, total_frames, height, width, channels, overlap):
        """Stitch temporal chunks back together with temporal fade blending"""
        result = torch.zeros((total_frames, height, width, channels), dtype=chunks[0].dtype, device=chunks[0].device)
        
        for i, ((t_start, t_end), chunk) in enumerate(zip(temporal_tiles, chunks)):
            chunk_frames = chunk.shape[0]
            
            if i == 0:
                result[t_start:t_start+chunk_frames] = chunk
            else:
                prev_end = temporal_tiles[i-1][1]
                overlap_start = max(t_start, prev_end - overlap)
                overlap_end = min(t_end, prev_end)
                
                if overlap_start < overlap_end:
                    overlap_frames = overlap_end - overlap_start
                    chunk_overlap_start = overlap_start - t_start
                    
                    existing_frames = result[overlap_start:overlap_end]
                    new_frames = chunk[chunk_overlap_start:chunk_overlap_start+overlap_frames]
                    
                    fade_mask = self._create_temporal_fade_mask(overlap_frames, existing_frames.dtype, existing_frames.device)
                    
                    blended_frames = existing_frames * (1.0 - fade_mask) + new_frames * fade_mask
                    result[overlap_start:overlap_end] = blended_frames
                    
                    if overlap_end < t_start + chunk_frames:
                        non_overlap_start = overlap_end
                        chunk_offset = non_overlap_start - t_start
                        result[non_overlap_start:t_start+chunk_frames] = chunk[chunk_offset:]
                else:
                    result[t_start:t_start+chunk_frames] = chunk
        
        return result
    
    def _create_temporal_fade_mask(self, overlap_frames, dtype, device):
        """Create a temporal fade mask for smooth frame transitions"""
        fade_values = torch.linspace(0.0, 1.0, overlap_frames, dtype=dtype, device=device)
        fade_mask = fade_values.view(overlap_frames, 1, 1, 1)
        return fade_mask
    
    def _generate_wanvideo_processing_summary(self, all_tiles, temporal_tiles, spatial_tiles_h, spatial_tiles_w, total_tiles):
        """Generate comprehensive summary of the WanVideo tiled processing"""
        successful_tiles = sum(1 for tile in all_tiles if hasattr(tile, 'processing_status') and tile.processing_status == 'success')
        failed_tiles = total_tiles - successful_tiles
        
        summary = f"=== DIMENSION-WISE TILED WANVIDEO VACE PROCESSING SUMMARY ===\n"
        summary += f"Total tiles processed: {total_tiles}\n"
        summary += f"Successful WanVideo tiles: {successful_tiles}\n"
        summary += f"Failed tiles: {failed_tiles}\n"
        summary += f"WanVideo success rate: {(successful_tiles/total_tiles)*100:.1f}%\n\n"
        
        # Temporal consistency chain information
        summary += f"TEMPORAL CONSISTENCY CHAIN:\n"
        summary += f"Temporal chunks: {len(temporal_tiles)}\n"
        for i, (start, end) in enumerate(temporal_tiles):
            if i == 0:
                ref_source = "User-provided reference images"
            else:
                ref_source = f"Frame references from Chunk {i-1} (frame_overlap-based)"
            summary += f"  Chunk {i+1}: frames {start}-{end-1} ({end-start} frames) - {ref_source}\n"
        
        summary += f"\nSpatial tiles per frame: {len(spatial_tiles_h)}×{len(spatial_tiles_w)}\n"
        summary += f"Height tiles: {len(spatial_tiles_h)}\n"
        for i, (start, end) in enumerate(spatial_tiles_h):
            summary += f"  Row {i+1}: pixels {start}-{end-1} ({end-start} pixels)\n"
        
        summary += f"\nWidth tiles: {len(spatial_tiles_w)}\n"
        for i, (start, end) in enumerate(spatial_tiles_w):
            summary += f"  Col {i+1}: pixels {start}-{end-1} ({end-start} pixels)\n"
        
        # Analyze temporal consistency chain usage
        chain_usage = {}
        for tile in all_tiles:
            if hasattr(tile, 'ref_frame_source'):
                source = tile.ref_frame_source
                if source not in chain_usage:
                    chain_usage[source] = 0
                chain_usage[source] += 1
        
        summary += f"\nTemporal reference chain usage:\n"
        for source, count in chain_usage.items():
            summary += f"  {source}: {count} tiles\n"
        
        # Group tiles by processing status
        if failed_tiles > 0:
            summary += f"\nFailed tiles details:\n"
            for tile in all_tiles:
                if hasattr(tile, 'processing_status') and tile.processing_status != 'success':
                    summary += f"  T{tile.temporal_index}:L{tile.line_index}:C{tile.column_index}: {tile.processing_status}\n"
        
        # Add dimension-wise processing information
        summary += f"\nDimension-wise processing algorithm with temporal consistency:\n"
        summary += f"  1. Temporal consistency chain: Each chunk uses references from previous chunk\n"
        summary += f"  2. WanVideo VACE processing: Each tile processed with spatial-specific references\n"
        summary += f"  3. Column-wise stitching: Vertical fade blending within temporal chunks\n"
        summary += f"  4. Line-wise stitching: Horizontal fade blending across columns\n"
        summary += f"  5. Temporal stitching: Frame-based fade blending across time\n"
        summary += f"  6. Cropping: Final video cropped to original input dimensions\n"
        
        summary += f"\nTemporal consistency ensures each tile position maintains coherent evolution across chunks\n"
        summary += f"Reference frame extracted from frame_overlap position for smooth temporal transitions\n"
        summary += f"Large video processed through {total_tiles} individual WanVideo VACE operations with temporal continuity\n"
        
        return summary

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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TiledWanImageToMask": ImageToMask,
    "TiledWanImageStatistics": ImageStatistics,
    "TiledWanMaskStatistics": MaskStatistics,
    "TileAndStitchBack": TileAndStitchBack,
    "TiledWanVideoVACEpipe": TiledWanVideoVACEpipe,
    "TiledWanInpaintCropImproved": InpaintCropImproved,
    "TiledWanInpaintStitchImproved": InpaintStitchImproved,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledWanImageToMask": "TiledWan Image To Mask",
    "TiledWanImageStatistics": "TiledWan Image Statistics",
    "TiledWanMaskStatistics": "TiledWan Mask Statistics",
    "TileAndStitchBack": "Tile and Stitch Back",
    "TiledWanVideoVACEpipe": "TiledWan Video VACE Pipeline",
    "TiledWanInpaintCropImproved": "TiledWan Inpaint Crop Improved",
    "TiledWanInpaintStitchImproved": "TiledWan Inpaint Stitch Improved"
}
