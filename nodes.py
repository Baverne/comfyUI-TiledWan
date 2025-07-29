import torch
import comfy.utils
import comfy.model_management
import node_helpers
import random
import os
import sys
import importlib
import traceback


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



class TiledWanVideoSamplerSimple:
    """
    Complete wrapper for WanVideoSampler that exposes all possible input arguments.
    This node provides maximum compatibility with the original WanVideoSampler by exposing
    all parameters exactly as they should be, with proper defaults (rope_function="comfy").
    TeaCache is controlled exclusively through cache_args input.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Complete WanVideoSampler inputs with all possible arguments
        """
        return {
            "required": {
                # Core WanVideoSampler inputs
                "model": ("WANVIDEOMODEL",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (["unipc", "unipc/beta", "dpm++", "dpm++/beta","dpm++_sde", "dpm++_sde/beta", "euler", "euler/beta", "euler/accvideo", "deis", "lcm", "lcm/beta", "flowmatch_causvid", "flowmatch_distill", "multitalk"],
                    {"default": 'unipc'}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "batched_cfg": ("BOOLEAN", {"default": False}),
                "rope_function": (["default", "comfy"], {"default": "comfy"}),
            },
            "optional": {
                # Optional core inputs
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "samples": ("LATENT",),
                
                # Advanced optional arguments
                "feta_args": ("FETAARGS",),
                "context_options": ("CONTEXTOPTIONS",),
                "cache_args": ("CACHEARGS",),
                "slg_args": ("SLGARGS",),
                "loop_args": ("LOOPARGS",),
                "experimental_args": ("EXPERIMENTALARGS",),
                "sigmas": ("SIGMAS",),
                "unianimate_poses": ("UNIANIMATE_POSES",),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS",),
                "uni3c_embeds": ("UNI3C_EMBEDS",),
                "multitalk_embeds": ("MULTITALK_EMBEDS",),
                "freeinit_args": ("FREEINIT_ARGS",),
                "teacache_args": ("TEACACHE_ARGS",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "process_sampler"
    OUTPUT_NODE = True
    CATEGORY = "TiledWan"

    def process_sampler(self, **kwargs):
        """
        Execute WanVideoSampler with all possible arguments
        """
        
        print("\n" + "="*80)
        print("                TILEDWAN WANVIDEO SAMPLER SIMPLE")
        print("="*80)
        print("🚀 Starting complete WanVideo sampler...")
        
        try:
            # Import the WanVideoSampler
            import sys
            import os
            import importlib
            
            custom_nodes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ComfyUI-WanVideoWrapper")
            if custom_nodes_path not in sys.path:
                sys.path.append(custom_nodes_path)
            
            # Add parent directory to sys.path and import normally
            parent_path = os.path.dirname(custom_nodes_path)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
            
            # Import the package normally to handle relative imports
            package_name = os.path.basename(custom_nodes_path)  # "ComfyUI-WanVideoWrapper"
            wanvideo_package = importlib.import_module(f"{package_name}.nodes")
            
            WanVideoSampler = wanvideo_package.WanVideoSampler
            
            print("🎯 Running WanVideoSampler with all parameters...")
            print(f"📊 Parameters received: {list(kwargs.keys())}")
            
            # Run the sampler with all arguments
            sampler_node = WanVideoSampler()
            latent_samples = sampler_node.process(
                model=kwargs["model"],
                image_embeds=kwargs["image_embeds"],
                steps=kwargs.get("steps", 30),
                cfg=kwargs.get("cfg", 6.0),
                shift=kwargs.get("shift", 5.0),
                seed=kwargs.get("seed", 0),
                scheduler=kwargs.get("scheduler", "unipc"),
                riflex_freq_index=kwargs.get("riflex_freq_index", 0),
                text_embeds=kwargs.get("text_embeds"),
                samples=kwargs.get("samples"),
                denoise_strength=kwargs.get("denoise_strength", 1.0),
                force_offload=kwargs.get("force_offload", True),
                
                # All the advanced arguments
                cache_args=kwargs.get("cache_args"),
                feta_args=kwargs.get("feta_args"),
                context_options=kwargs.get("context_options"),
                flowedit_args=None,  # Not in the list but exists in the original call
                batched_cfg=kwargs.get("batched_cfg", False),
                slg_args=kwargs.get("slg_args"),
                rope_function=kwargs.get("rope_function", "comfy"),
                loop_args=kwargs.get("loop_args"),
                experimental_args=kwargs.get("experimental_args"),
                sigmas=kwargs.get("sigmas"),
                unianimate_poses=kwargs.get("unianimate_poses"),
                fantasytalking_embeds=kwargs.get("fantasytalking_embeds"),
                uni3c_embeds=kwargs.get("uni3c_embeds"),
                multitalk_embeds=kwargs.get("multitalk_embeds"),
                freeinit_args=kwargs.get("freeinit_args")
            )[0]
            
            print("✅ Complete WanVideo sampler completed successfully!")
            print(f"📤 Output shape: {latent_samples.get('samples', 'Unknown').shape if hasattr(latent_samples.get('samples', None), 'shape') else 'No shape info'}")
            print(f"🔧 Cache used: {'Yes' if kwargs.get('cache_args') else 'No'}")
            print(f"🔧 SLG used: {'Yes' if kwargs.get('slg_args') else 'No'}")
            print(f"🔧 Experimental used: {'Yes' if kwargs.get('experimental_args') else 'No'}")
            print(f"🔧 Rope function: {kwargs.get('rope_function', 'comfy')}")
            print("="*80 + "\n")
            
            return (latent_samples,)
            
        except Exception as e:
            import traceback
            print(f"❌ Error in complete WanVideo sampler: {str(e)}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            print("="*80 + "\n")
            
            # Return dummy output in case of error
            dummy_latent = {"samples": torch.zeros((1, 16, 10, 8, 8))}  # Dummy latent
            
            return (dummy_latent,)


class TiledWanVideoSLGSimple:
    """
    A very simple test node that only wraps WanVideoSLG for debugging.
    This is the simplest possible wrapper to test import and basic functionality.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Basic WanVideoSLG inputs only
        """
        return {
            "required": {
                "blocks": ("STRING", {"default": "10"}),
                "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SLGARGS",)
    RETURN_NAMES = ("slg_args",)
    FUNCTION = "process_slg"
    OUTPUT_NODE = True
    CATEGORY = "TiledWan"

    def process_slg(self, **kwargs):
        """
        Execute only the WanVideoSLG
        """
        
        print("\n" + "="*80)
        print("                TILEDWAN WANVIDEO SLG SIMPLE")
        print("="*80)
        print("🚀 Starting simple WanVideoSLG test...")
        
        try:
            # Import the WanVideoSLG
            import sys
            import os
            import importlib
            import traceback
            custom_nodes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ComfyUI-WanVideoWrapper")
            print(f"🔍 Looking for WanVideo nodes in: {custom_nodes_path}")
            
            if custom_nodes_path not in sys.path:
                sys.path.append(custom_nodes_path)
                print(f"📁 Added to sys.path: {custom_nodes_path}")
            
            # Test if the path exists
            if os.path.exists(custom_nodes_path):
                print(f"✅ Path exists: {custom_nodes_path}")
                nodes_file = os.path.join(custom_nodes_path, "nodes.py")
                if os.path.exists(nodes_file):
                    print(f"✅ nodes.py found: {nodes_file}")
                else:
                    print(f"❌ nodes.py NOT found: {nodes_file}")
            else:
                print(f"❌ Path does NOT exist: {custom_nodes_path}")
            
            print("🔄 Attempting import...")
            
            # Solution: Add parent directory to sys.path and import normally
            parent_path = os.path.dirname(custom_nodes_path)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
                print(f"📁 Added parent to sys.path: {parent_path}")
            
            # Import the package normally to handle relative imports
            import importlib
            package_name = os.path.basename(custom_nodes_path)  # "ComfyUI-WanVideoWrapper"
            wanvideo_package = importlib.import_module(f"{package_name}.nodes")
            
            WanVideoSLG = wanvideo_package.WanVideoSLG
            print("✅ Import successful!")
            
            print("🎯 Running WanVideoSLG with basic parameters...")
            print(f"📊 Parameters received: {list(kwargs.keys())}")
            
            # Run the SLG node
            slg_node = WanVideoSLG()
            slg_args = slg_node.process(
                blocks=kwargs.get("blocks", "10"),
                start_percent=kwargs.get("start_percent", 0.1),
                end_percent=kwargs.get("end_percent", 1.0)
            )[0]
            
            print("✅ Simple WanVideoSLG completed successfully!")
            print(f"📤 Output type: {type(slg_args)}")
            print(f"📤 Output content: {slg_args}")
            print("="*80 + "\n")
            
            return (slg_args,)
            
        except Exception as e:
            import traceback
            print(f"❌ Error in simple WanVideoSLG: {str(e)}")
            print(f"📋 Full traceback:")
            print(traceback.format_exc())
            print("="*80 + "\n")
            
            # Return dummy output in case of error
            dummy_slg = {"blocks": "10", "start_percent": 0.1, "end_percent": 1.0}
            
            return (dummy_slg,)


class WanVideoVACEpipe:
    """
    Complete WanVideo pipeline that encapsulates WanVideo nodes:
    - WanVideoVACEEncode -> image_embeds  
    - WanVideoSampler (original, not TiledWan wrapper)
    - WanVideoDecode -> final video output
    
    This node provides a complete video generation pipeline in a single node.
    TeaCache, SLG, and Experimental arguments are provided externally for better modularity.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Streamlined pipeline inputs with external args for TeaCache, SLG, and Experimental features
        """
        return {
            "required": {
                # Core pipeline inputs
                "model": ("WANVIDEOMODEL",),
                "vae": ("WANVAE",),
                
                # WanVideoSampler core inputs
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (["unipc", "unipc/beta", "dpm++", "dpm++/beta","dpm++_sde", "dpm++_sde/beta", "euler", "euler/beta", "euler/accvideo", "deis", "lcm", "lcm/beta", "flowmatch_causvid", "flowmatch_distill", "multitalk"],
                    {"default": 'unipc'}),
                
                # WanVideoVACEEncode inputs
                "vace_width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8}),
                "vace_height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8}),
                "vace_num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4}),
                "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # WanVideoDecode inputs
                "decode_enable_vae_tiling": ("BOOLEAN", {"default": False}),
                "decode_tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8}),
                "decode_tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8}),
                "decode_tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8}),
                "decode_tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8}),
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
                "vace_input_frames": ("IMAGE",),
                "vace_ref_images": ("IMAGE",),
                "vace_input_masks": ("MASK",),
                "vace_tiled_vae": ("BOOLEAN", {"default": False}),
                
                # WanVideoDecode optional inputs
                "decode_normalization": (["default", "minmax"], {"default": "default"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("video", "latents")
    FUNCTION = "process_wanvideo_pipeline"
    OUTPUT_NODE = True
    CATEGORY = "TiledWan"

    def process_wanvideo_pipeline(self, **kwargs):
        """
        Execute the streamlined WanVideo pipeline using external arguments
        """
        
        print("\n" + "="*80)
        print("                    WANVIDEO VACE PIPELINE")
        print("="*80)
        print("🚀 Starting streamlined WanVideo VACE pipeline...")
        print(f"📊 Total parameters received: {len(kwargs)}")
        
        try:
            # Import WanVideo nodes we need
            import sys
            import os
            import importlib
            import traceback
            
            custom_nodes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ComfyUI-WanVideoWrapper")
            if custom_nodes_path not in sys.path:
                sys.path.append(custom_nodes_path)
                print(f"📁 Added to sys.path: {custom_nodes_path}")
            
            # Add parent directory to sys.path and import normally
            parent_path = os.path.dirname(custom_nodes_path)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
                print(f"📁 Added parent to sys.path: {parent_path}")
            
            # Import the package normally to handle relative imports
            package_name = os.path.basename(custom_nodes_path)  # "ComfyUI-WanVideoWrapper"
            print(f"🔄 Importing WanVideo package: {package_name}")
            wanvideo_package = importlib.import_module(f"{package_name}.nodes")
            
            WanVideoSampler = wanvideo_package.WanVideoSampler
            WanVideoVACEEncode = wanvideo_package.WanVideoVACEEncode
            WanVideoDecode = wanvideo_package.WanVideoDecode
            print("✅ Required WanVideo nodes imported successfully!")
            
            # Get external arguments directly from inputs
            cache_args = kwargs.get("cache_args")
            slg_args = kwargs.get("slg_args")
            experimental_args = kwargs.get("experimental_args")
            
            print(f"📦 External arguments provided:")
            print(f"   • TeaCache args: {'Yes' if cache_args else 'No'}")
            print(f"   • SLG args: {'Yes' if slg_args else 'No'}")
            print(f"   • Experimental args: {'Yes' if experimental_args else 'No'}")
            
            # Step 1: Create VACE embeds
            print("📦 Step 1: Creating VACE embeds...")
            vace_node = WanVideoVACEEncode()
            vace_embeds = vace_node.process(
                vae=kwargs["vae"],
                width=kwargs.get("vace_width", 832),
                height=kwargs.get("vace_height", 480),
                num_frames=kwargs.get("vace_num_frames", 81),
                strength=kwargs.get("vace_strength", 1.0),
                vace_start_percent=kwargs.get("vace_start_percent", 0.0),
                vace_end_percent=kwargs.get("vace_end_percent", 1.0),
                input_frames=kwargs.get("vace_input_frames"),
                ref_images=kwargs.get("vace_ref_images"),
                input_masks=kwargs.get("vace_input_masks"),
                tiled_vae=kwargs.get("vace_tiled_vae", False)
            )[0]
            print(f"✅ VACE embeds created: {type(vace_embeds)}")
            
            # Step 2: Run WanVideoSampler
            print("🎯 Step 2: Running WanVideoSampler...")
            print(f"   • Model: {type(kwargs['model'])}")
            print(f"   • VACE embeds: {type(vace_embeds)}")
            print(f"   • Steps: {kwargs.get('steps', 30)}")
            print(f"   • CFG: {kwargs.get('cfg', 6.0)}")
            print(f"   • Shift: {kwargs.get('shift', 5.0)}")
            print(f"   • Seed: {kwargs.get('seed', 0)}")
            print(f"   • Scheduler: {kwargs.get('scheduler', 'unipc')}")
            print(f"   • Rope function: {kwargs.get('rope_function', 'comfy')}")
            print(f"   • External TeaCache: {'Yes' if cache_args else 'No'}")
            print(f"   • External SLG: {'Yes' if slg_args else 'No'}")
            print(f"   • External Experimental: {'Yes' if experimental_args else 'No'}")
            
            sampler_node = WanVideoSampler()
            latent_samples = sampler_node.process(
                model=kwargs["model"],
                image_embeds=vace_embeds,  # VACE embeds -> image_embeds
                steps=kwargs.get("steps", 30),
                cfg=kwargs.get("cfg", 6.0),
                shift=kwargs.get("shift", 5.0),
                seed=kwargs.get("seed", 0),
                scheduler=kwargs.get("scheduler", "unipc"),
                riflex_freq_index=kwargs.get("riflex_freq_index", 0),
                text_embeds=kwargs.get("text_embeds"),
                samples=kwargs.get("samples"),
                denoise_strength=kwargs.get("denoise_strength", 1.0),
                force_offload=kwargs.get("force_offload", True),
                
                # External arguments (much cleaner!)
                cache_args=cache_args,
                slg_args=slg_args,
                experimental_args=experimental_args,
                
                # Other WanVideoSampler arguments
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
            
            print("✅ WanVideoSampler completed!")
            print(f"   • Output type: {type(latent_samples)}")
            if hasattr(latent_samples, 'get') and 'samples' in latent_samples:
                print(f"   • Latent shape: {latent_samples['samples'].shape}")
            
            # Step 3: Decode latents to video
            print("🎬 Step 3: Decoding latents to video...")
            print(f"   • VAE: {type(kwargs['vae'])}")
            print(f"   • Latents: {type(latent_samples)}")
            print(f"   • Enable VAE tiling: {kwargs.get('decode_enable_vae_tiling', False)}")
            print(f"   • Tile size: {kwargs.get('decode_tile_x', 272)}x{kwargs.get('decode_tile_y', 272)}")
            print(f"   • Normalization: {kwargs.get('decode_normalization', 'default')}")
            
            decode_node = WanVideoDecode()
            video_output = decode_node.decode(
                vae=kwargs["vae"],
                samples=latent_samples,  # Sampler samples -> decode samples
                enable_vae_tiling=kwargs.get("decode_enable_vae_tiling", False),
                tile_x=kwargs.get("decode_tile_x", 272),
                tile_y=kwargs.get("decode_tile_y", 272),
                tile_stride_x=kwargs.get("decode_tile_stride_x", 144),
                tile_stride_y=kwargs.get("decode_tile_stride_y", 128),
                normalization=kwargs.get("decode_normalization", "default")
            )[0]
            
            print(f"✅ Video decoding completed!")
            print(f"   • Video output type: {type(video_output)}")
            if hasattr(video_output, 'shape'):
                print(f"   • Video shape: {video_output.shape}")
            
            print("🎉 Streamlined WanVideo VACE pipeline finished successfully!")
            print("   Pipeline used external TeaCache, SLG, and Experimental arguments for maximum modularity")
            print("="*80 + "\n")
            
            return (video_output, latent_samples)
            
        except Exception as e:
            import traceback
            print(f"❌ Error in WanVideo VACE pipeline: {str(e)}")
            print(f"📋 Full traceback:")
            print(traceback.format_exc())
            print("="*80 + "\n")
            
            # Return dummy outputs in case of error
            dummy_video = torch.zeros((1, 64, 64, 3))  # Dummy video frame
            dummy_latent = {"samples": torch.zeros((1, 16, 10, 8, 8))}  # Dummy latent
            
            return (dummy_video, dummy_latent)


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
                    # Last chunk with large overlap - use full region fade blending
                    print(f"🔥 Last chunk: Large overlap detected ({actual_overlap} > {frame_overlap})")
                    
                    # Calculate overlap region
                    overlap_frames = actual_overlap
                    
                    # Get regions for blending
                    result_overlap = result[current_t-overlap_frames:current_t]
                    chunk_overlap = chunk_content[:overlap_frames]
                    
                    # Create temporal fade mask
                    fade_mask = self._create_temporal_fade_mask(overlap_frames, result_overlap.dtype, result_overlap.device)
                    
                    # Apply fade blending across the entire overlap region
                    blended_overlap = result_overlap * (1.0 - fade_mask) + chunk_overlap * fade_mask
                    result[current_t-overlap_frames:current_t] = blended_overlap
                    
                    # Place the non-overlapping part
                    if chunk_frames > overlap_frames:
                        result[current_t:current_t+chunk_frames-overlap_frames] = chunk_content[overlap_frames:]
                    current_t += chunk_frames - overlap_frames
                else:
                    # Normal overlap handling
                    overlap_frames = min(frame_overlap, chunk_frames // 2, current_t)
                    
                    if overlap_frames > 0:
                        # Get regions for blending
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
    5. Large overlap handling: Last tiles in each dimension use full-region fade blending
    6. Cropping: Final video cropped to exact input dimensions
    7. Memory management: Proper model offloading between tiles to prevent leaks
    
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
                "video": ("IMAGE",),
                "mask": ("MASK",),
                
                # Core WanVideo pipeline inputs
                "model": ("WANVIDEOMODEL",),
                "vae": ("WANVAE",),
                
                # Tiling parameters
                "target_frames": ("INT", {"default": 81, "min": 16, "max": 200, "step": 1}),
                "target_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "target_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "frame_overlap": ("INT", {"default": 10, "min": 0, "max": 40, "step": 1}),
                "spatial_overlap": ("INT", {"default": 20, "min": 0, "max": 100, "step": 4}),
                
                # WanVideoSampler core parameters
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (["unipc", "unipc/beta", "dpm++", "dpm++/beta","dpm++_sde", "dpm++_sde/beta", "euler", "euler/beta", "euler/accvideo", "deis", "lcm", "lcm/beta", "flowmatch_causvid", "flowmatch_distill", "multitalk"],
                    {"default": 'unipc'}),
                
                # WanVideoVACEEncode parameters  
  
  
                "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
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
    FUNCTION = "process_tiled_wanvideo"
    OUTPUT_NODE = True
    CATEGORY = "TiledWan"

    def process_tiled_wanvideo(self, **kwargs):
        """
        Execute the streamlined WanVideo pipeline using external arguments
        """
        
        print("\n" + "="*80)
        print("                    WANVIDEO VACE PIPELINE")
        print("="*80)
        print("🚀 Starting streamlined WanVideo VACE processing...")
        print(f"📊 Total parameters received: {len(kwargs)}")
        
        try:
            # Import WanVideo nodes we need
            import sys
            import os
            import importlib
            import traceback
            
            custom_nodes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ComfyUI-WanVideoWrapper")
            if custom_nodes_path not in sys.path:
                sys.path.append(custom_nodes_path)
                print(f"📁 Added to sys.path: {custom_nodes_path}")
            
            # Add parent directory to sys.path and import normally
            parent_path = os.path.dirname(custom_nodes_path)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
                print(f"📁 Added parent to sys.path: {parent_path}")
            
            # Import the package normally to handle relative imports
            package_name = os.path.basename(custom_nodes_path)  # "ComfyUI-WanVideoWrapper"
            print(f"🔄 Importing WanVideo package: {package_name}")
            wanvideo_package = importlib.import_module(f"{package_name}.nodes")
            
            WanVideoSampler = wanvideo_package.WanVideoSampler
            WanVideoVACEEncode = wanvideo_package.WanVideoVACEEncode
            WanVideoDecode = wanvideo_package.WanVideoDecode
            print("✅ WanVideo nodes imported successfully!")
            
            # Input validation
            batch_size, height, width, channels = video.shape
            print(f"📹 Input video shape: {video.shape} (B×H×W×C)")
            print(f"🎯 Target tile size: {target_frames} frames × {target_width}×{target_height}")
            print(f"🔗 Overlaps: {frame_overlap} frames, {spatial_overlap} pixels")
            
            # Calculate temporal chunks
            temporal_tiles = self._calculate_temporal_tiles(batch_size, target_frames, frame_overlap)
            print(f"⏱️  Temporal chunks: {len(temporal_tiles)}")
            
            # Show temporal chunk layout for verification
            print(f"📅 Temporal chunk layout:")
            for i, (t_start, t_end) in enumerate(temporal_tiles):
                overlap_info = ""
                if i > 0:
                    prev_end = temporal_tiles[i-1][1]
                    actual_overlap = prev_end - t_start
                    overlap_info = f" (overlap: {actual_overlap} frames with previous chunk)"
                print(f"   Chunk {i+1}: frames {t_start}-{t_end-1}{overlap_info}")
            
            # Initialize variables for temporal processing
            completed_chunks = []
            current_ref_image = kwargs.get("vace_ref_images")  # Use provided ref for first chunk
            total_successful_tiles = 0
            total_failed_tiles = 0
            
            # TEMPORAL-FIRST PROCESSING: Process each temporal chunk completely before moving to next
            for temporal_idx, (t_start, t_end) in enumerate(temporal_tiles):
                print(f"\n🎬 TEMPORAL CHUNK {temporal_idx + 1}/{len(temporal_tiles)} (frames {t_start}-{t_end-1})")
                
                # Calculate reference frame logic for this chunk
                if temporal_idx == 0:
                    # First chunk: use provided reference or None
                    ref_info = "User-provided reference" if current_ref_image is not None else "No reference"
                    print(f"   🔗 Reference: {ref_info}")
                else:
                    # For subsequent chunks: use the frame from previous chunk that corresponds to t_start
                    prev_chunk = completed_chunks[temporal_idx - 1]
                    prev_t_start, prev_t_end = temporal_tiles[temporal_idx - 1]
                    
                    # Calculate which frame from previous chunk corresponds to current chunk's start
                    frame_offset_in_prev_chunk = t_start - prev_t_start
                    
                    if frame_offset_in_prev_chunk < prev_chunk.shape[0]:
                        # Use the corresponding frame from previous chunk
                        current_ref_image = prev_chunk[frame_offset_in_prev_chunk:frame_offset_in_prev_chunk+1]
                        print(f"   🔗 Reference: Frame {frame_offset_in_prev_chunk} from previous chunk (global frame {t_start})")
                    else:
                        # Fallback to last frame of previous chunk if offset is out of bounds
                        current_ref_image = prev_chunk[-1:]
                        print(f"   🔗 Reference: Last frame from previous chunk (fallback)")
            
                try:
                    # Extract temporal chunk
                    video_chunk = video[t_start:t_end]
                    mask_chunk = mask[t_start:t_end]
                    
                    # Process this temporal chunk completely (spatial tiling + stitching)
                    processed_chunk, chunk_stats = self._process_temporal_chunk_complete(
                        video_chunk, mask_chunk, current_ref_image, model, vae,
                        target_width, target_height, spatial_overlap, 
                        WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                        steps, cfg, shift, seed + temporal_idx, scheduler,
                        vace_strength, vace_start_percent, vace_end_percent,
                        decode_enable_vae_tiling, decode_tile_x, decode_tile_y,
                        decode_tile_stride_x, decode_tile_stride_y,
                        force_offload_between_tiles, debug_mode, kwargs
                    )
                    
                    completed_chunks.append(processed_chunk)
                    total_successful_tiles += chunk_stats['successful']
                    total_failed_tiles += chunk_stats['failed']
                    
                    print(f"   ✅ Temporal chunk {temporal_idx + 1} completed: {processed_chunk.shape}")
                    
                    # Force memory cleanup after each temporal chunk
                    if force_offload_between_tiles:
                        self._force_memory_cleanup(model, vae)
                        
                except Exception as chunk_error:
                    print(f"   ❌ Error processing temporal chunk {temporal_idx + 1}: {str(chunk_error)}")
                    print(f"   🔄 Using original video frames as fallback for chunk {temporal_idx + 1}")
                    
                    # Fallback to original video chunk
                    fallback_chunk = video[t_start:t_end]
                    completed_chunks.append(fallback_chunk)
                    total_failed_tiles += self._estimate_tiles_in_chunk(fallback_chunk.shape, target_width, target_height, spatial_overlap)
        
        # FINAL TEMPORAL STITCHING: Combine all completed temporal chunks
        print(f"\n🔗 FINAL TEMPORAL STITCHING: Combining {len(completed_chunks)} temporal chunks...")
        if len(completed_chunks) == 1:
            final_video = completed_chunks[0]
        else:
            final_video = self._stitch_temporal_chunks_final(completed_chunks, temporal_tiles, frame_overlap)
        
        # Crop to original dimensions
        print(f"\n✂️ CROPPING TO ORIGINAL DIMENSIONS...")
        print(f"   📐 Stitched video shape: {final_video.shape}")
        print(f"   🎯 Target shape: {video.shape}")
        final_video = final_video[:batch_size, :height, :width, :channels]
        print(f"   ✂️ Final video shape: {final_video.shape}")
        
        # Generate processing summary
        total_tiles = total_successful_tiles + total_failed_tiles
        success_rate = (total_successful_tiles / total_tiles * 100) if total_tiles > 0 else 0
        
        summary = f"=== TEMPORAL-FIRST TILED WANVIDEO PROCESSING SUMMARY ===\n"
        summary += f"Temporal chunks processed: {len(temporal_tiles)}\n"
        summary += f"Total tiles processed: {total_tiles}\n"
        summary += f"Successful tiles: {total_successful_tiles}\n"
        summary += f"Failed tiles: {total_failed_tiles}\n"
        summary += f"Success rate: {success_rate:.1f}%\n"
        summary += f"Temporal consistency: Enhanced through sequential processing\n"
        summary += f"Reference propagation: Corresponding overlapping frame → Next chunk\n"
        summary += f"Processing order: Temporal-first → Spatial tiling per chunk\n"
        summary += f"Final video dimensions: {final_video.shape}"
        
        print(f"✅ Temporal-first tiled WanVideo processing completed!")
        print(f"📈 Success rate: {success_rate:.1f}% ({total_successful_tiles}/{total_tiles} tiles)")
        print(f"🔗 Temporal consistency enhanced through proper reference frame propagation")
        print("="*80 + "\n")
        
        return (final_video, summary)
        
    except Exception as e:
        print(f"❌ Critical error in temporal-first processing: {str(e)}")
        import traceback
        print(f"📋 Full traceback:")
        print(traceback.format_exc())
        print("="*80 + "\n")
        
        error_summary = f"Critical error in temporal-first processing: {str(e)}"
        return (video, error_summary)
    

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TiledWanImageToMask": ImageToMask,
    "TiledWanImageStatistics": ImageStatistics,
    "TiledWanMaskStatistics": MaskStatistics,
    "TiledWanVideoSamplerSimple": TiledWanVideoSamplerSimple,
    "TiledWanVideoSLGSimple": TiledWanVideoSLGSimple,
    "WanVideoVACEpipe": WanVideoVACEpipe,
    "TileAndStitchBack": TileAndStitchBack,
    "TiledWanVideoVACEpipe": TiledWanVideoVACEpipe
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledWanImageToMask": "TiledWan Image To Mask",
    "TiledWanImageStatistics": "TiledWan Image Statistics",
    "TiledWanMaskStatistics": "TiledWan Mask Statistics",
    "TiledWanVideoSamplerSimple": "TiledWan Video Sampler Simple",
    "TiledWanVideoSLGSimple": "TiledWan Video SLG Simple",
    "WanVideoVACEpipe": "WanVideo VACE Pipeline",
    "TileAndStitchBack": "Tile and Stitch Back",
    "TiledWanVideoVACEpipe": "Tiled WanVideo VACE Pipeline"
}
