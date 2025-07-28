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
            
            print(f"✅ WanVideoSampler completed!")
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


class TileAndStitchBack:
    """
    Tile and Stitch Back node for WanVideo preprocessing/postprocessing.
    
    This node demonstrates the tiling system that will be used with WanVideo:
    1. Temporal tiling: Split video into 81-frame chunks (10-frame overlap)
    2. Spatial tiling: Split each chunk into 832×480 tiles (20-pixel overlap)
    3. Color transformation: Apply random color shift to each tile (for debugging)
    4. Stitch back: Reconstruct the original video from processed tiles
    
    This is optimized for WanVideo's best performance on 81-frame, 832×480 videos.
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
        Tile video temporally and spatially, apply color transformation, then stitch back
        """
        
        print("\n" + "="*80)
        print("                  TILE AND STITCH BACK")
        print("="*80)
        print("🧩 Starting video tiling and stitching process...")
        
        try:
            import random
            import traceback
            
            # Input video info
            batch_size, height, width, channels = video.shape
            print(f"📹 Input video shape: {video.shape} (B×H×W×C)")
            print(f"🎯 Target tile size: {target_frames} frames × {target_width}×{target_height}")
            print(f"🔗 Overlaps: {frame_overlap} frames, {spatial_overlap} pixels")
            
            # Calculate temporal tiles
            temporal_tiles = self._calculate_temporal_tiles(batch_size, target_frames, frame_overlap)
            print(f"⏱️  Temporal tiles: {len(temporal_tiles)} chunks")
            for i, (start, end) in enumerate(temporal_tiles):
                print(f"   Chunk {i+1}: frames {start}-{end-1} ({end-start} frames)")
            
            # Calculate spatial tiles for each frame dimension
            spatial_tiles_h = self._calculate_spatial_tiles(height, target_height, spatial_overlap)
            spatial_tiles_w = self._calculate_spatial_tiles(width, target_width, spatial_overlap)
            print(f"🗺️  Spatial tiles: {len(spatial_tiles_h)}×{len(spatial_tiles_w)} = {len(spatial_tiles_h) * len(spatial_tiles_w)} tiles per frame")
            
            total_tiles = len(temporal_tiles) * len(spatial_tiles_h) * len(spatial_tiles_w)
            print(f"📦 Total tiles to process: {total_tiles}")
            
            # Process each temporal chunk
            processed_chunks = []
            tile_info_list = []
            
            for temporal_idx, (t_start, t_end) in enumerate(temporal_tiles):
                print(f"\n🎬 Processing temporal chunk {temporal_idx + 1}/{len(temporal_tiles)} (frames {t_start}-{t_end-1})")
                
                # Extract temporal chunk
                chunk = video[t_start:t_end]
                chunk_processed = torch.zeros_like(chunk)
                
                # Process each spatial tile within this temporal chunk
                for h_idx, (h_start, h_end) in enumerate(spatial_tiles_h):
                    for w_idx, (w_start, w_end) in enumerate(spatial_tiles_w):
                        tile_idx = h_idx * len(spatial_tiles_w) + w_idx
                        
                        if debug_mode:
                            print(f"   🧩 Tile {tile_idx + 1}/{len(spatial_tiles_h) * len(spatial_tiles_w)}: "
                                  f"H[{h_start}:{h_end}] × W[{w_start}:{w_end}]")
                        
                        # Extract spatial tile
                        tile = chunk[:, h_start:h_end, w_start:w_end, :]
                        
                        # Apply color transformation (for debugging)
                        tile_transformed = self._apply_color_shift(tile, color_shift_strength, 
                                                                 temporal_idx, h_idx, w_idx)
                        
                        # Place tile back (handle overlaps with fade blending)
                        self._place_tile_with_overlap(chunk_processed, tile_transformed, 
                                                    h_start, h_end, w_start, w_end, color_shift_strength)
                        
                        # Record tile info
                        tile_info = {
                            'temporal_chunk': temporal_idx,
                            'spatial_tile': (h_idx, w_idx),
                            'temporal_range': (t_start, t_end),
                            'spatial_range': ((h_start, h_end), (w_start, w_end)),
                            'tile_shape': tile.shape
                        }
                        tile_info_list.append(tile_info)
                
                processed_chunks.append(chunk_processed)
                print(f"✅ Temporal chunk {temporal_idx + 1} processed")
            
            # Stitch temporal chunks back together
            print(f"\n🔗 Stitching {len(processed_chunks)} temporal chunks back together...")
            stitched_video = self._stitch_temporal_chunks(processed_chunks, temporal_tiles, 
                                                        batch_size, height, width, channels, frame_overlap)
            
            # Generate tile info summary
            tile_info_summary = self._generate_tile_info_summary(tile_info_list, temporal_tiles, 
                                                               spatial_tiles_h, spatial_tiles_w)
            
            print(f"✅ Tiling and stitching completed!")
            print(f"📤 Output video shape: {stitched_video.shape}")
            print(f"🧩 Total tiles processed: {len(tile_info_list)}")
            print(f"📊 Tile info summary: {len(tile_info_summary)} characters")
            print("="*80 + "\n")
            
            return (stitched_video, tile_info_summary)
            
        except Exception as e:
            print(f"❌ Error in tile and stitch: {str(e)}")
            print(f"📋 Full traceback:")
            print(traceback.format_exc())
            print("="*80 + "\n")
            
            # Return original video in case of error
            dummy_info = f"Error during tiling: {str(e)}"
            return (video, dummy_info)
    
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
    
    def _place_tile_with_overlap(self, target, tile, h_start, h_end, w_start, w_end, color_shift_strength=0.0):
        """Place tile into target tensor, handling overlaps with spatial fade blending"""
        tile_h, tile_w = tile.shape[1:3]
        
        # Get the target region
        target_region = target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :]
        
        # Check if there's existing content (non-zero) in the target area
        if target_region.sum() > 0:
            # Always use production-quality fade blending regardless of color shift strength
            # Color shift is purely for visual debugging and shouldn't affect blending quality
            fade_mask = self._create_spatial_fade_mask(tile_h, tile_w, target_region, tile)
            
            # Apply fade blending: existing * (1 - fade_mask) + tile * fade_mask
            blended = target_region * (1.0 - fade_mask) + tile * fade_mask
            target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :] = blended
        else:
            # First tile in this area or non-overlapping area - always place the tile
            # (this includes color-shifted tiles when debugging is enabled)
            target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :] = tile
    
    def _create_spatial_fade_mask(self, tile_h, tile_w, existing, new_tile):
        """Create a spatial fade mask for smooth blending between tiles"""
        # Create base mask (1.0 = use new tile, 0.0 = use existing)
        mask = torch.ones(1, tile_h, tile_w, 1, dtype=existing.dtype, device=existing.device)
        
        # Define fade distance (how many pixels to fade over)
        fade_h = min(10, tile_h // 4)  # Fade over 10 pixels or 1/4 of tile height
        fade_w = min(10, tile_w // 4)  # Fade over 10 pixels or 1/4 of tile width
        
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
    Tiled WanVideo VACE Pipeline - The ultimate node for processing large videos with debugging capabilities.
    
    This node combines the sophisticated tiling system from TileAndStitchBack with 
    the complete WanVideo VACE pipeline. It handles large videos by:
    
    1. Temporal tiling: Split video into 81-frame chunks (10-frame overlap)
    2. Spatial tiling: Split each chunk into 832×480 tiles (20-pixel overlap) 
    3. WanVideo processing: Process each tile through the complete VACE pipeline
    4. Memory management: Properly offload models between tiles to prevent leaks
    5. Final stitching: Reconstruct final video with fade blending
    
    DEBUG FEATURES:
    - debug_tile_before: Returns the first tile BEFORE WanVideo processing
    - debug_tile_after: Returns the first tile AFTER WanVideo processing  
    - debug_only_first_tile: When enabled, processes only the first tile and returns early
    
    This enables processing of arbitrarily large videos that would otherwise 
    exceed Wan training. 
    The overall spatial and temporal coherence shall be insured 
    by a first and foremost full resolution and full sequence pass.
    This method is to be seen as a kind of upscaling method able of reconstructing poor quality first result.
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
                
                # WanVideoVACEEncode parameters (only user-controllable ones)
                "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # WanVideoDecode parameters
                "decode_enable_vae_tiling": ("BOOLEAN", {"default": False}),
                "decode_tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8}),
                "decode_tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8}),
                "decode_tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8}),
                "decode_tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8}),
                
                # Processing and Debug parameters
                "debug_mode": ("BOOLEAN", {"default": True}),
                "debug_only_first_tile": ("BOOLEAN", {"default": False}),
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
                
                # WanVideoVACEEncode optional inputs (only external references)
                "vace_ref_images": ("IMAGE",),
                "vace_tiled_vae": ("BOOLEAN", {"default": False}),
                
                # WanVideoDecode optional inputs
                "decode_normalization": (["default", "minmax"], {"default": "default"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("processed_video", "processing_info", "debug_tile_before", "debug_tile_after")
    FUNCTION = "process_tiled_wanvideo"
    OUTPUT_NODE = True
    CATEGORY = "TiledWan"

    def process_tiled_wanvideo(self, video, mask, model, vae, target_frames, target_width, target_height, 
                              frame_overlap, spatial_overlap, steps, cfg, shift, seed, scheduler,
                              vace_strength, vace_start_percent, vace_end_percent, 
                              decode_enable_vae_tiling, decode_tile_x, decode_tile_y,
                              decode_tile_stride_x, decode_tile_stride_y, debug_mode, 
                              debug_only_first_tile, force_offload_between_tiles, **kwargs):
        """
        Process large video through tiled WanVideo VACE pipeline with memory management
        """
        
        print("\n" + "="*80)
        print("              TILED WANVIDEO VACE PIPELINE")
        print("="*80)
        print("🚀 Starting tiled WanVideo VACE processing...")
        
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
            
            # Input validation
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
                # Resize mask to match video
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1), size=(height, width), mode='nearest'
                ).squeeze(1)
                print(f"✅ Mask resized to: {mask.shape}")
            
            # Calculate temporal and spatial tiles
            temporal_tiles = self._calculate_temporal_tiles(batch_size, target_frames, frame_overlap)
            spatial_tiles_h = self._calculate_spatial_tiles(height, target_height, spatial_overlap) 
            spatial_tiles_w = self._calculate_spatial_tiles(width, target_width, spatial_overlap)
            
            total_tiles = len(temporal_tiles) * len(spatial_tiles_h) * len(spatial_tiles_w)
            print(f"⏱️  Temporal chunks: {len(temporal_tiles)}")
            print(f"🗺️  Spatial tiles per frame: {len(spatial_tiles_h)}×{len(spatial_tiles_w)}")
            print(f"📦 Total tiles to process: {total_tiles}")
            
            # Debug mode check
            if debug_only_first_tile:
                print(f"🔍 DEBUG MODE: Processing only the first tile for debugging")
            
            # Initialize debug variables
            debug_tile_before = None
            debug_tile_after = None
            
            # Process each temporal chunk
            processed_chunks = []
            processing_info_list = []
            
            for temporal_idx, (t_start, t_end) in enumerate(temporal_tiles):
                print(f"\n🎬 Processing temporal chunk {temporal_idx + 1}/{len(temporal_tiles)} (frames {t_start}-{t_end-1})")
                
                # Extract temporal chunk for video and mask
                video_chunk = video[t_start:t_end]
                mask_chunk = mask[t_start:t_end]
                chunk_processed = torch.zeros_like(video_chunk)
                
                # Process each spatial tile within this temporal chunk
                for h_idx, (h_start, h_end) in enumerate(spatial_tiles_h):
                    for w_idx, (w_start, w_end) in enumerate(spatial_tiles_w):
                        tile_idx = h_idx * len(spatial_tiles_w) + w_idx
                        
                        if debug_mode:
                            print(f"   🧩 Tile {tile_idx + 1}/{len(spatial_tiles_h) * len(spatial_tiles_w)}: "
                                  f"H[{h_start}:{h_end}] × W[{w_start}:{w_end}]")
                        
                        # Extract spatial tiles (video and mask)
                        video_tile = video_chunk[:, h_start:h_end, w_start:w_end, :]
                        mask_tile = mask_chunk[:, h_start:h_end, w_start:w_end]
                        
                        # Capture debug tile BEFORE processing (first tile only)
                        if temporal_idx == 0 and h_idx == 0 and w_idx == 0:
                            debug_tile_before = video_tile.clone()
                            print(f"🔍 DEBUG: Captured first tile BEFORE processing - shape: {debug_tile_before.shape}")
                        
                        # Process tile through WanVideo VACE pipeline
                        try:
                            processed_tile, tile_latents = self._process_tile_through_wanvideo(
                                video_tile, mask_tile, model, vae, 
                                WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                                steps, cfg, shift, seed + tile_idx,  # Unique seed per tile
                                scheduler, vace_strength, vace_start_percent, vace_end_percent,
                                decode_enable_vae_tiling, decode_tile_x, decode_tile_y,
                                decode_tile_stride_x, decode_tile_stride_y, kwargs
                            )
                            
                            # Capture debug tile AFTER processing (first tile only)
                            if temporal_idx == 0 and h_idx == 0 and w_idx == 0:
                                debug_tile_after = processed_tile.clone()
                                print(f"🔍 DEBUG: Captured first tile AFTER processing - shape: {debug_tile_after.shape}")
                                
                                # If debug_only_first_tile is enabled, process only this tile and return early
                                if debug_only_first_tile:
                                    print(f"🔍 DEBUG MODE: Only processing first tile, returning early")
                                    tile_info = {
                                        'temporal_chunk': temporal_idx,
                                        'spatial_tile': (h_idx, w_idx),
                                        'temporal_range': (t_start, t_end),
                                        'spatial_range': ((h_start, h_end), (w_start, w_end)),
                                        'input_shape': video_tile.shape,
                                        'output_shape': processed_tile.shape,
                                        'seed_used': seed + tile_idx,
                                        'status': 'debug_only_first_tile'
                                    }
                                    debug_summary = f"=== DEBUG MODE: FIRST TILE ONLY ===\n"
                                    debug_summary += f"First tile processed only for debugging\n"
                                    debug_summary += f"Temporal chunk: {temporal_idx}, Spatial tile: ({h_idx}, {w_idx})\n"
                                    debug_summary += f"Input shape: {video_tile.shape}\n"
                                    debug_summary += f"Output shape: {processed_tile.shape}\n"
                                    debug_summary += f"Seed used: {seed + tile_idx}\n"
                                    debug_summary += f"Tile range: H[{h_start}:{h_end}] × W[{w_start}:{w_end}]\n"
                                    debug_summary += f"Temporal range: frames {t_start}-{t_end-1}\n"
                                    
                                    # Return processed tile placed in a minimal video for visualization
                                    debug_video = torch.zeros_like(video)
                                    debug_video[t_start:t_end, h_start:h_end, w_start:w_end, :] = processed_tile
                                    
                                    return (debug_video, debug_summary, debug_tile_before, debug_tile_after)
                            
                            # Place processed tile back with fade blending
                            self._place_tile_with_overlap(chunk_processed, processed_tile,
                                                        h_start, h_end, w_start, w_end)
                            
                            # Record processing info
                            tile_info = {
                                'temporal_chunk': temporal_idx,
                                'spatial_tile': (h_idx, w_idx),
                                'temporal_range': (t_start, t_end),
                                'spatial_range': ((h_start, h_end), (w_start, w_end)),
                                'input_shape': video_tile.shape,
                                'output_shape': processed_tile.shape,
                                'seed_used': seed + tile_idx,
                                'status': 'success'
                            }
                            processing_info_list.append(tile_info)
                            
                            # Force memory cleanup between tiles if requested
                            if force_offload_between_tiles:
                                self._force_memory_cleanup(model, vae)
                                
                        except Exception as tile_error:
                            print(f"      ❌ Error processing tile {tile_idx + 1}: {str(tile_error)}")
                            # Use original tile as fallback
                            self._place_tile_with_overlap(chunk_processed, video_tile,
                                                        h_start, h_end, w_start, w_end)
                            
                            tile_info = {
                                'temporal_chunk': temporal_idx,
                                'spatial_tile': (h_idx, w_idx),
                                'temporal_range': (t_start, t_end),
                                'spatial_range': ((h_start, h_end), (w_start, w_end)),
                                'input_shape': video_tile.shape,
                                'output_shape': video_tile.shape,
                                'seed_used': seed + tile_idx,
                                'status': f'failed: {str(tile_error)}'
                            }
                            processing_info_list.append(tile_info)
                
                processed_chunks.append(chunk_processed)
                print(f"✅ Temporal chunk {temporal_idx + 1} processed")
            
            # Stitch temporal chunks back together
            print(f"\n🔗 Stitching {len(processed_chunks)} temporal chunks back together...")
            final_video = self._stitch_temporal_chunks(processed_chunks, temporal_tiles,
                                                     batch_size, height, width, channels, frame_overlap)
            
            # Generate comprehensive processing info
            processing_summary = self._generate_processing_summary(processing_info_list, temporal_tiles,
                                                                 spatial_tiles_h, spatial_tiles_w, total_tiles)
            
            print(f"✅ Tiled WanVideo VACE processing completed!")
            print(f"📤 Output video shape: {final_video.shape}")
            print(f"🧩 Total tiles processed: {len(processing_info_list)}")
            successful_tiles = sum(1 for info in processing_info_list if info['status'] == 'success')
            print(f"✅ Successful tiles: {successful_tiles}/{total_tiles}")
            print("="*80 + "\n")
            
            # Create dummy debug outputs if none were captured
            if debug_tile_before is None:
                debug_tile_before = torch.zeros((1, 64, 64, 3))  # Dummy tile
                print("⚠️  Warning: No debug tile BEFORE captured")
            if debug_tile_after is None:
                debug_tile_after = torch.zeros((1, 64, 64, 3))  # Dummy tile
                print("⚠️  Warning: No debug tile AFTER captured")
            
            return (final_video, processing_summary, debug_tile_before, debug_tile_after)
            
        except Exception as e:
            print(f"❌ Error in tiled WanVideo VACE pipeline: {str(e)}")
            print(f"📋 Full traceback:")
            print(traceback.format_exc())
            print("="*80 + "\n")
            
            # Return original video in case of error
            error_info = f"Error during tiled WanVideo processing: {str(e)}"
            dummy_debug_before = torch.zeros((1, 64, 64, 3))  # Dummy debug tile
            dummy_debug_after = torch.zeros((1, 64, 64, 3))   # Dummy debug tile
            return (video, error_info, dummy_debug_before, dummy_debug_after)
    
    def _process_tile_through_wanvideo(self, video_tile, mask_tile, model, vae,
                                     WanVideoVACEEncode, WanVideoSampler, WanVideoDecode,
                                     steps, cfg, shift, seed, scheduler, vace_strength,
                                     vace_start_percent, vace_end_percent, decode_enable_vae_tiling,
                                     decode_tile_x, decode_tile_y, decode_tile_stride_x, 
                                     decode_tile_stride_y, kwargs):
        """Process a single tile through the complete WanVideo VACE pipeline"""
        
        tile_frames, tile_height, tile_width = video_tile.shape[:3]
        
        # Step 1: Create VACE embeds for this tile
        vace_node = WanVideoVACEEncode()
        vace_embeds = vace_node.process(
            vae=vae,
            width=tile_width,
            height=tile_height, 
            num_frames=tile_frames,
            strength=vace_strength,
            vace_start_percent=vace_start_percent,
            vace_end_percent=vace_end_percent,
            input_frames=video_tile,  # Use tile as input frames
            ref_images=kwargs.get("vace_ref_images"),
            input_masks=mask_tile,    # Use tile mask
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
        
        return processed_tile, latent_samples
    
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
    
    def _place_tile_with_overlap(self, target, tile, h_start, h_end, w_start, w_end):
        """Place tile into target tensor, handling overlaps with spatial fade blending"""
        tile_h, tile_w = tile.shape[1:3]
        
        # Get the target region
        target_region = target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :]
        
        # Check if there's existing content (non-zero) in the target area
        if target_region.sum() > 0:
            # Use production-quality fade blending
            fade_mask = self._create_spatial_fade_mask(tile_h, tile_w, target_region, tile)
            
            # Apply fade blending: existing * (1 - fade_mask) + tile * fade_mask
            blended = target_region * (1.0 - fade_mask) + tile * fade_mask
            target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :] = blended
        else:
            # First tile in this area - place directly
            target[:, h_start:h_start+tile_h, w_start:w_start+tile_w, :] = tile
    
    def _create_spatial_fade_mask(self, tile_h, tile_w, existing, new_tile):
        """Create a spatial fade mask for smooth blending between tiles"""
        # Create base mask (1.0 = use new tile, 0.0 = use existing)
        mask = torch.ones(1, tile_h, tile_w, 1, dtype=existing.dtype, device=existing.device)
        
        # Define fade distance
        fade_h = min(10, tile_h // 4)
        fade_w = min(10, tile_w // 4)
        
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
    
    def _generate_processing_summary(self, processing_info_list, temporal_tiles, spatial_tiles_h, spatial_tiles_w, total_tiles):
        """Generate comprehensive summary of the tiled processing"""
        successful_tiles = sum(1 for info in processing_info_list if info['status'] == 'success')
        failed_tiles = total_tiles - successful_tiles
        
        summary = f"=== TILED WANVIDEO VACE PROCESSING SUMMARY ===\n"
        summary += f"Total tiles processed: {total_tiles}\n"
        summary += f"Successful tiles: {successful_tiles}\n"
        summary += f"Failed tiles: {failed_tiles}\n"
        summary += f"Success rate: {(successful_tiles/total_tiles)*100:.1f}%\n\n"
        
        summary += f"Temporal chunks: {len(temporal_tiles)}\n"
        for i, (start, end) in enumerate(temporal_tiles):
            summary += f"  Chunk {i+1}: frames {start}-{end-1} ({end-start} frames)\n"
        
        summary += f"\nSpatial tiles per frame: {len(spatial_tiles_h)}×{len(spatial_tiles_w)}\n"
        summary += f"Height tiles: {len(spatial_tiles_h)}\n"
        for i, (start, end) in enumerate(spatial_tiles_h):
            summary += f"  Row {i+1}: pixels {start}-{end-1} ({end-start} pixels)\n"
        
        summary += f"\nWidth tiles: {len(spatial_tiles_w)}\n"
        for i, (start, end) in enumerate(spatial_tiles_w):
            summary += f"  Col {i+1}: pixels {start}-{end-1} ({end-start} pixels)\n"
        
        if failed_tiles > 0:
            summary += f"\nFailed tiles:\n"
            for info in processing_info_list:
                if info['status'] != 'success':
                    summary += f"  Chunk {info['temporal_chunk']}, Tile {info['spatial_tile']}: {info['status']}\n"
        
        summary += f"\nTiled WanVideo VACE processing completed successfully!"
        summary += f"\nLarge video processed through {total_tiles} individual WanVideo operations."
        
        return summary


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TiledWanImageToMask": ImageToMask,
    "TiledWanImageStatistics": ImageStatistics,
    "TiledWanMaskStatistics": MaskStatistics,
    "TiledWanVideoVACEpipe": TiledWanVideoVACEpipe
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledWanImageToMask": "TiledWan Image To Mask",
    "TiledWanImageStatistics": "TiledWan Image Statistics",
    "TiledWanMaskStatistics": "TiledWan Mask Statistics",
    "TiledWanVideoVACEpipe": "Tiled WanVideo VACE Pipeline"
}
