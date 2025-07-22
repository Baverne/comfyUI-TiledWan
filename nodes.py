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


class TiledWanVideoSamplerTestConcat:
    """
    A test node that concatenates and executes a WanVideo pipeline by chaining:
    WanVideoSampler -> WanVideoTeaCache -> WanVideoVACEEncode -> WanVideoSLG -> WanVideoExperimentalArgs -> WanVideoDecode
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Combine all input types from the WanVideo nodes we want to chain
        """
        return {
            "required": {
                # Core WanVideoSampler inputs
                "model": ("WANVIDEOMODEL",),
                "vae": ("WANVAE",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (["unipc", "unipc/beta", "dpm++", "dpm++/beta","dpm++_sde", "dpm++_sde/beta", "euler", "euler/beta", "euler/accvideo", "deis", "lcm", "lcm/beta", "flowmatch_causvid", "flowmatch_distill", "multitalk"],
                    {"default": 'unipc'}),
                
                # TeaCache inputs
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.001}),
                "teacache_start_step": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1}),
                "teacache_end_step": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1}),
                "teacache_use_coefficients": ("BOOLEAN", {"default": True}),
                
                # VACE Encode inputs
                "vace_width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8}),
                "vace_height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8}),
                "vace_num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4}),
                "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # SLG inputs
                "slg_blocks": ("STRING", {"default": "10"}),
                "slg_start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "slg_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Experimental Args inputs
                "exp_video_attention_split_steps": ("STRING", {"default": ""}),
                "exp_cfg_zero_star": ("BOOLEAN", {"default": False}),
                "exp_use_zero_init": ("BOOLEAN", {"default": False}),
                "exp_zero_star_steps": ("INT", {"default": 0, "min": 0}),
                "exp_use_fresca": ("BOOLEAN", {"default": False}),
                "exp_fresca_scale_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "exp_fresca_scale_high": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.01}),
                "exp_fresca_freq_cutoff": ("INT", {"default": 20, "min": 0, "max": 10000, "step": 1}),
                
                # Decode inputs
                "decode_enable_vae_tiling": ("BOOLEAN", {"default": False}),
                "decode_tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8}),
                "decode_tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8}),
                "decode_tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8}),
                "decode_tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8}),
            },
            "optional": {
                # Optional inputs from various nodes
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "samples": ("LATENT",),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "teacache_cache_device": (["main_device", "offload_device"], {"default": "offload_device"}),
                "teacache_mode": (["e", "e0"], {"default": "e"}),
                "vace_input_frames": ("IMAGE",),
                "vace_ref_images": ("IMAGE",),
                "vace_input_masks": ("MASK",),
                "vace_tiled_vae": ("BOOLEAN", {"default": False}),
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
        Execute the complete WanVideo pipeline by chaining the nodes
        """
        
        print("\n" + "="*80)
        print("                    TILEDWAN WANVIDEO PIPELINE")
        print("="*80)
        print("🚀 Starting WanVideo pipeline execution...")
        
        try:
            # Import the WanVideo nodes we need
            import sys
            import os
            import importlib
            
            custom_nodes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ComfyUI-WanVideoWrapper")
            sys.path.append(custom_nodes_path)
            
            # Add parent directory to sys.path and import normally
            parent_path = os.path.dirname(custom_nodes_path)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
            
            # Import the package normally to handle relative imports
            package_name = os.path.basename(custom_nodes_path)  # "ComfyUI-WanVideoWrapper"
            wanvideo_package = importlib.import_module(f"{package_name}.nodes")
            
            WanVideoSampler = wanvideo_package.WanVideoSampler
            WanVideoTeaCache = wanvideo_package.WanVideoTeaCache
            WanVideoVACEEncode = wanvideo_package.WanVideoVACEEncode
            WanVideoSLG = wanvideo_package.WanVideoSLG
            WanVideoExperimentalArgs = wanvideo_package.WanVideoExperimentalArgs
            WanVideoDecode = wanvideo_package.WanVideoDecode
            
            # Step 1: Create TeaCache arguments
            print("📦 Step 1: Preparing TeaCache arguments...")
            cache_device = kwargs.get("teacache_cache_device", "offload_device")
            if cache_device == "main_device":
                import comfy.model_management as mm
                cache_device = mm.get_torch_device()
            else:
                import comfy.model_management as mm
                cache_device = mm.unet_offload_device()
            
            teacache_args = {
                "cache_type": "TeaCache",
                "rel_l1_thresh": kwargs.get("teacache_rel_l1_thresh", 0.3),
                "start_step": kwargs.get("teacache_start_step", 1),
                "end_step": kwargs.get("teacache_end_step", -1),
                "cache_device": cache_device,
                "use_coefficients": kwargs.get("teacache_use_coefficients", True),
                "mode": kwargs.get("teacache_mode", "e"),
            }
            
            # Step 2: Create SLG arguments
            print("📦 Step 2: Preparing SLG arguments...")
            slg_node = WanVideoSLG()
            slg_args = slg_node.process(
                kwargs.get("slg_blocks", "10"),
                kwargs.get("slg_start_percent", 0.1),
                kwargs.get("slg_end_percent", 1.0)
            )[0]
            
            # Step 3: Create Experimental arguments
            print("📦 Step 3: Preparing Experimental arguments...")
            exp_node = WanVideoExperimentalArgs()
            exp_args = exp_node.process(
                video_attention_split_steps=kwargs.get("exp_video_attention_split_steps", ""),
                cfg_zero_star=kwargs.get("exp_cfg_zero_star", False),
                use_zero_init=kwargs.get("exp_use_zero_init", False),
                zero_star_steps=kwargs.get("exp_zero_star_steps", 0),
                use_fresca=kwargs.get("exp_use_fresca", False),
                fresca_scale_low=kwargs.get("exp_fresca_scale_low", 1.0),
                fresca_scale_high=kwargs.get("exp_fresca_scale_high", 1.25),
                fresca_freq_cutoff=kwargs.get("exp_fresca_freq_cutoff", 20)
            )[0]
            
            # Step 4: Create VACE embeds if needed (this might be created before sampling)
            print("📦 Step 4: Preparing VACE embeds...")
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
            
            # Step 5: Run the sampler with all arguments
            print("🎯 Step 5: Running WanVideo Sampler...")
            sampler_node = WanVideoSampler()
            latent_samples = sampler_node.process(
                model=kwargs["model"],
                image_embeds=vace_embeds,  # Use the VACE embeds we created
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
                cache_args=teacache_args,
                slg_args=slg_args,
                experimental_args=exp_args
            )[0]
            
            # Step 6: Decode the latents to video
            print("🎬 Step 6: Decoding latents to video...")
            decode_node = WanVideoDecode()
            video_output = decode_node.decode(
                vae=kwargs["vae"],
                samples=latent_samples,
                enable_vae_tiling=kwargs.get("decode_enable_vae_tiling", False),
                tile_x=kwargs.get("decode_tile_x", 272),
                tile_y=kwargs.get("decode_tile_y", 272),
                tile_stride_x=kwargs.get("decode_tile_stride_x", 144),
                tile_stride_y=kwargs.get("decode_tile_stride_y", 128),
                normalization=kwargs.get("decode_normalization", "default")
            )[0]
            
            print("✅ WanVideo pipeline completed successfully!")
            print("="*80 + "\n")
            
            return (video_output, latent_samples)
            
        except Exception as e:
            print(f"❌ Error in WanVideo pipeline: {str(e)}")
            print("="*80 + "\n")
            
            # Return dummy outputs in case of error
            dummy_video = torch.zeros((1, 64, 64, 3))  # Dummy video frame
            dummy_latent = {"samples": torch.zeros((1, 16, 10, 8, 8))}  # Dummy latent
            
            return (dummy_video, dummy_latent)


class TiledWanVideoSamplerSimple:
    """
    A simple test node that only wraps the WanVideoSampler for easier debugging.
    This node exposes only the essential WanVideoSampler parameters without chaining other nodes.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Basic WanVideoSampler inputs only
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
            },
            "optional": {
                # Optional inputs
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "samples": ("LATENT",),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "process_sampler"
    OUTPUT_NODE = True
    CATEGORY = "TiledWan"

    def process_sampler(self, **kwargs):
        """
        Execute only the WanVideoSampler
        """
        
        print("\n" + "="*80)
        print("                TILEDWAN WANVIDEO SAMPLER SIMPLE")
        print("="*80)
        print("🚀 Starting simple WanVideo sampler...")
        
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
            
            print("🎯 Running WanVideoSampler with basic parameters...")
            print(f"📊 Parameters received: {list(kwargs.keys())}")
            
            # Run the sampler with basic arguments (no cache, no slg, no experimental)
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
                # Optional args set to None for simplicity
                cache_args=None,
                feta_args=None,
                context_options=None,
                flowedit_args=None,
                batched_cfg=False,
                slg_args=None,
                rope_function="default",
                loop_args=None,
                experimental_args=None,
                sigmas=None,
                unianimate_poses=None,
                fantasytalking_embeds=None,
                uni3c_embeds=None,
                multitalk_embeds=None,
                freeinit_args=None
            )[0]
            
            print("✅ Simple WanVideo sampler completed successfully!")
            print(f"📤 Output shape: {latent_samples.get('samples', 'Unknown').shape if hasattr(latent_samples.get('samples', None), 'shape') else 'No shape info'}")
            print("="*80 + "\n")
            
            return (latent_samples,)
            
        except Exception as e:
            import traceback
            print(f"❌ Error in simple WanVideo sampler: {str(e)}")
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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TiledWanImageToMask": ImageToMask,
    "TiledWanImageStatistics": ImageStatistics,
    "TiledWanMaskStatistics": MaskStatistics,
    "TiledWanVideoSamplerTestConcat": TiledWanVideoSamplerTestConcat,
    "TiledWanVideoSamplerSimple": TiledWanVideoSamplerSimple,
    "TiledWanVideoSLGSimple": TiledWanVideoSLGSimple
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledWanImageToMask": "TiledWan Image To Mask",
    "TiledWanImageStatistics": "TiledWan Image Statistics",
    "TiledWanMaskStatistics": "TiledWan Mask Statistics",
    "TiledWanVideoSamplerTestConcat": "TiledWan Video Sampler Test Concat",
    "TiledWanVideoSamplerSimple": "TiledWan Video Sampler Simple",
    "TiledWanVideoSLGSimple": "TiledWan Video SLG Simple"
}
