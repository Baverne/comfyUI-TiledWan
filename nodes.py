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


class ImageStatistics:
    """
    Calculate and display comprehensive statistics for an image.
    Shows min, max, mean, variance, median, and other useful statistics.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "show_per_channel": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("image", "min_value", "max_value", "mean_value", "variance", "median_value")
    FUNCTION = "calculate_statistics"
    OUTPUT_NODE = True  # This allows the node to display output in the console
    CATEGORY = "TiledWan"

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
                "mask": ("MASK",),
                "analyze_connected_components": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
            },
        }

    RETURN_TYPES = ("MASK", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("mask", "min_value", "max_value", "mean_value", "variance", "median_value", "white_pixel_count")
    FUNCTION = "calculate_mask_statistics"
    OUTPUT_NODE = True  # This allows the node to display output in the console
    CATEGORY = "TiledWan"

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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TiledWanImageToMask": ImageToMask,
    "TiledWanImageStatistics": ImageStatistics,
    "TiledWanMaskStatistics": MaskStatistics
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledWanImageToMask": "TiledWan Image To Mask",
    "TiledWanImageStatistics": "TiledWan Image Statistics",
    "TiledWanMaskStatistics": "TiledWan Mask Statistics"
}
