# TiledWan ComfyUI Node Set

A custom node set for ComfyUI that provides tiled processing capabilities.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone <your-repo-url> comfyUI-TiledWan
```

2. Restart ComfyUI

## Nodes

### TiledWan Image To Mask
Converts an image to a mask by extracting a specific color channel (red, green, blue, or alpha) with optional clamping and normalization.

**Inputs:**
- `image`: IMAGE - Input image to convert to mask
- `channel`: COMBO - Channel to extract ("red", "green", "blue", "alpha")
- `clamp_output`: BOOLEAN - Whether to clamp output values between 0 and 1 (default: True)
- `normalize_output`: BOOLEAN - Whether to normalize output to 0-1 range (default: True)

**Outputs:**
- `mask`: MASK - The extracted channel as a mask

**Features:**
- Supports RGBA images (if alpha channel is requested but doesn't exist, returns a fully opaque mask)
- Optional normalization: scales the mask values to use the full 0-1 range
- Optional clamping: ensures all values are between 0 and 1
- Robust handling of different image formats

### TiledWan Image Statistics
Analyze images and display comprehensive statistics including min, max, mean, variance, median and distribution analysis.

**Inputs:**
- `image`: IMAGE - Input image to analyze
- `show_per_channel`: BOOLEAN - Whether to show per-channel statistics for RGB/RGBA images (default: True)

**Outputs:**
- `image`: IMAGE - The original image (pass-through)
- `min_value`: FLOAT - Minimum pixel value in the image
- `max_value`: FLOAT - Maximum pixel value in the image
- `mean_value`: FLOAT - Average pixel value
- `variance`: FLOAT - Variance of pixel values
- `median_value`: FLOAT - Median pixel value

**Features:**
- Comprehensive statistics: min, max, mean, median, variance, standard deviation
- Per-channel analysis for multi-channel images
- Value distribution histogram (10 bins)
- Percentile analysis (1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th)
- Memory usage and image format information
- Automatic issue detection (negative values, out-of-range values, constant images)
- Console output with detailed formatting for easy reading

### TiledWan Mask Statistics
Analyze masks and display comprehensive statistics including coverage, density, connected components and distribution analysis.

**Inputs:**
- `mask`: MASK - Input mask to analyze
- `analyze_connected_components`: BOOLEAN - Whether to perform connected components analysis (default: True)

**Outputs:**
- `mask`: MASK - The original mask (pass-through)
- `min_value`: FLOAT - Minimum pixel value in the mask
- `max_value`: FLOAT - Maximum pixel value in the mask
- `mean_value`: FLOAT - Average pixel value
- `variance`: FLOAT - Variance of pixel values
- `median_value`: FLOAT - Median pixel value
- `white_pixel_count`: INT - Number of pixels considered "white" (>0.5)

**Features:**
- Standard statistics: min, max, mean, median, variance, standard deviation
- Mask-specific analysis: white/black pixel counts, coverage percentage, density classification
- Value distribution: exact counts for few unique values, histogram for complex masks
- Connected components analysis: estimated number and size of separate mask regions
- Mask quality assessment: binary mask detection, uniformity analysis
- Automatic issue detection: negative values, out-of-range values, sparse/dense masks

### TiledWan Video Sampler Test Concat
A comprehensive pipeline node that chains multiple WanVideo operations into a single node. This is a test/development node that combines WanVideoSampler, WanVideoTeaCache, WanVideoVACEEncode, WanVideoSLG, WanVideoExperimentalArgs, and WanVideoDecode for streamlined video generation.

**Inputs (Required):**
- `model`: WANVIDEOMODEL - The WanVideo model to use for generation
- `vae`: WANVAE - The VAE for encoding/decoding video latents
- `image_embeds`: WANVIDIMAGE_EMBEDS - Image embeddings for the video
- `steps`: INT - Number of sampling steps (default: 30)
- `cfg`: FLOAT - Classifier-free guidance scale (default: 6.0) 
- `shift`: FLOAT - Shift parameter for scheduler (default: 5.0)
- `seed`: INT - Random seed for generation
- `scheduler`: COMBO - Sampling scheduler type (unipc, euler, dpm++, etc.)

**TeaCache Parameters:**
- `teacache_rel_l1_thresh`: FLOAT - TeaCache threshold for caching decisions (default: 0.3)
- `teacache_start_step`: INT - Step to start applying TeaCache (default: 1)
- `teacache_end_step`: INT - Step to end applying TeaCache (default: -1)
- `teacache_use_coefficients`: BOOLEAN - Use calculated coefficients for accuracy (default: True)

**VACE Encode Parameters:**
- `vace_width`: INT - Video width in pixels (default: 832)
- `vace_height`: INT - Video height in pixels (default: 480)
- `vace_num_frames`: INT - Number of frames to generate (default: 81)
- `vace_strength`: FLOAT - VACE encoding strength (default: 1.0)
- `vace_start_percent`: FLOAT - Start percentage for VACE application (default: 0.0)
- `vace_end_percent`: FLOAT - End percentage for VACE application (default: 1.0)

**SLG Parameters:**
- `slg_blocks`: STRING - Transformer blocks to skip unconditioned guidance on (default: "10")
- `slg_start_percent`: FLOAT - Start percentage for SLG application (default: 0.1)
- `slg_end_percent`: FLOAT - End percentage for SLG application (default: 1.0)

**Experimental Parameters:**
- `exp_video_attention_split_steps`: STRING - Steps to split video attention for multiple prompts
- `exp_cfg_zero_star`: BOOLEAN - Enable CFG Zero Star optimization (default: False)
- `exp_use_zero_init`: BOOLEAN - Use zero initialization (default: False)
- `exp_zero_star_steps`: INT - Number of zero star steps (default: 0)
- `exp_use_fresca`: BOOLEAN - Enable FreSca frequency scaling (default: False)
- `exp_fresca_scale_low`: FLOAT - FreSca low frequency scale (default: 1.0)
- `exp_fresca_scale_high`: FLOAT - FreSca high frequency scale (default: 1.25)
- `exp_fresca_freq_cutoff`: INT - FreSca frequency cutoff (default: 20)

**Decode Parameters:**
- `decode_enable_vae_tiling`: BOOLEAN - Enable VAE tiling for memory efficiency (default: False)
- `decode_tile_x`: INT - VAE tile width in pixels (default: 272)
- `decode_tile_y`: INT - VAE tile height in pixels (default: 272)
- `decode_tile_stride_x`: INT - VAE tile stride X (default: 144)
- `decode_tile_stride_y`: INT - VAE tile stride Y (default: 128)

**Optional Inputs:**
- `text_embeds`: WANVIDEOTEXTEMBEDS - Text embeddings for guided generation
- `samples`: LATENT - Initial latents for video-to-video processing
- `denoise_strength`: FLOAT - Denoising strength for v2v (default: 1.0)
- `vace_input_frames`: IMAGE - Input frames for VACE conditioning
- `vace_ref_images`: IMAGE - Reference images for VACE
- `vace_input_masks`: MASK - Input masks for VACE
- And many more optional parameters for fine control...

**Outputs:**
- `video`: IMAGE - Generated video frames as image sequence
- `latents`: LATENT - The generated latent representation

**Features:**
- **Complete Pipeline**: Integrates the entire WanVideo generation stack in one node
- **TeaCache Optimization**: Automatic caching for faster inference
- **VACE Encoding**: Advanced video-aware conditional encoding
- **SLG Guidance**: Selective layer guidance for improved quality
- **Experimental Features**: Access to cutting-edge techniques like FreSca and CFG Zero Star
- **Memory Efficient**: VAE tiling options for high-resolution generation
- **Progress Tracking**: Detailed console output showing pipeline progress
- **Error Handling**: Graceful fallback with dummy outputs if errors occur
- **Flexible Parameters**: Comprehensive control over every aspect of generation

## Usage

### TiledWan Image To Mask
The TiledWan Image To Mask node can be found in the "TiledWan" category in the ComfyUI node browser. 

1. Connect an image to the input
2. Select which channel to extract (red, green, blue, or alpha)
3. Configure clamping and normalization options as needed
4. The node will output a mask based on the selected channel

**Tips:**
- Use normalization when you want to maximize contrast in the resulting mask
- Use clamping to ensure compatibility with other ComfyUI nodes that expect 0-1 values
- The alpha channel option works even with RGB images (will create a fully opaque mask)

### TiledWan Image Statistics
The TiledWan Image Statistics node can be found in the "TiledWan" category in the ComfyUI node browser.

1. Connect any image to the input
2. Enable or disable per-channel analysis based on your needs
3. Run the workflow - statistics will be displayed in the ComfyUI console
4. The node also outputs individual statistic values that can be used by other nodes

**Console Output includes:**
- **Basic Info**: Image dimensions, total pixels, memory usage, data type
- **Global Statistics**: Min, max, mean, median, variance, standard deviation
- **Per-Channel Analysis**: Individual statistics for R, G, B, A channels
- **Distribution Analysis**: 10-bin histogram showing value distribution
- **Percentiles**: Key percentile values for detailed distribution analysis
- **Issue Detection**: Automatic warnings for potential problems

**Use Cases:**
- **Quality Control**: Detect images with unusual value ranges or distributions
- **Debugging**: Understand what's happening to your images in complex workflows
- **Optimization**: Identify images that might need preprocessing or normalization
- **Analysis**: Extract numerical statistics for further processing or decision making

**Tips:**
- Enable per-channel analysis for color images to understand channel balance
- Use the output values in mathematical nodes for dynamic workflow control
- Check the console output for warnings about potential image issues
- The pass-through design allows you to insert this node anywhere without breaking your workflow

### TiledWan Mask Statistics
The TiledWan Mask Statistics node can be found in the "TiledWan" category in the ComfyUI node browser.

1. Connect any mask to the input
2. Enable or disable connected components analysis based on your needs
3. Run the workflow - statistics will be displayed in the ComfyUI console
4. The node also outputs individual statistic values and white pixel count for further processing

**Console Output includes:**
- **Basic Info**: Mask dimensions, total pixels, memory usage, data type
- **Global Statistics**: Min, max, mean, median, variance, standard deviation
- **Mask-Specific Analysis**: White/black pixel counts, coverage percentage, density classification
- **Value Distribution**: Exact counts for simple masks or histogram for complex ones
- **Connected Components**: Estimated number and size of separate mask regions
- **Quality Assessment**: Binary mask detection, uniformity warnings

**Use Cases:**
- **Mask Quality Control**: Verify mask coverage and density
- **Segmentation Analysis**: Understand mask complexity and component distribution  
- **Workflow Optimization**: Use coverage data to skip processing on empty masks
- **Debugging**: Identify issues with mask generation or processing
- **Data Analysis**: Extract numerical mask properties for decision making

**Tips:**
- Enable connected components analysis for segmentation masks to understand object count
- Use the white_pixel_count output to conditionally process based on mask coverage
- Perfect binary masks (0 and 1 only) will be automatically detected and reported
- The pass-through design allows insertion anywhere in mask processing workflows
- Check console warnings for potential mask quality issues

### TiledWan Video Sampler Simple
A comprehensive wrapper for WanVideoSampler that exposes all possible input arguments for maximum flexibility and control.

**Inputs:**
**Required:**
- `model`: WANVIDEOMODEL - The WanVideo model for generation
- `image_embeds`: WANVIDIMAGE_EMBEDS - Image embeddings for conditioning
- `steps`: INT - Number of sampling steps (default: 30)
- `cfg`: FLOAT - Classifier-free guidance scale (default: 6.0, range: 0-30)
- `shift`: FLOAT - Shift parameter for sampling (default: 5.0, range: 0-1000)
- `seed`: INT - Random seed for reproducibility (default: 0)
- `scheduler`: COMBO - Sampling scheduler (default: "unipc")
- `riflex_freq_index`: INT - Riflex frequency index (default: 0)
- `denoise_strength`: FLOAT - Denoising strength (default: 1.0, range: 0-1)
- `force_offload`: BOOLEAN - Force model offloading (default: True)
- `batched_cfg`: BOOLEAN - Use batched CFG (default: False)
- `rope_function`: COMBO - RoPE function type (default: "default")

**Optional Advanced Arguments:**
- `text_embeds`: WANVIDEOTEXTEMBEDS - Text embeddings for conditioning
- `samples`: LATENT - Input latent samples for img2img
- `feta_args`: FETAARGS - FETA optimization arguments
- `context_options`: CONTEXTOPTIONS - Context handling options
- `cache_args`: CACHEARGS - Pre-built cache arguments
- `slg_args`: SLGARGS - SLG (Sparse Local Guidance) arguments
- `loop_args`: LOOPARGS - Loop generation arguments
- `experimental_args`: EXPERIMENTALARGS - Experimental features
- `sigmas`: SIGMAS - Custom noise schedule
- `unianimate_poses`: UNIANIMATE_POSES - UniAnimate pose data
- `fantasytalking_embeds`: FANTASYTALKING_EMBEDS - Fantasy talking embeddings
- `uni3c_embeds`: UNI3C_EMBEDS - Uni3C embeddings
- `multitalk_embeds`: MULTITALK_EMBEDS - MultiTalk embeddings
- `freeinit_args`: FREEINIT_ARGS - FreeInit arguments
- `teacache_args`: TEACACHE_ARGS - Pre-built TeaCache arguments

**TeaCache Manual Configuration:**
- `enable_teacache`: BOOLEAN - Enable TeaCache optimization (default: False)
- `teacache_rel_l1_thresh`: FLOAT - L1 threshold for cache (default: 0.3)
- `teacache_start_step`: INT - Cache start step (default: 1)
- `teacache_end_step`: INT - Cache end step (default: -1)
- `teacache_use_coefficients`: BOOLEAN - Use cache coefficients (default: True)
- `teacache_cache_device`: COMBO - Cache device ("main_device", "offload_device")
- `teacache_mode`: COMBO - Cache mode ("e", "e0")

**Outputs:**
- `latents`: LATENT - Generated latent samples

**Features:**
- **Complete Parameter Exposure**: All WanVideoSampler arguments are available
- **Flexible Caching**: Use pre-built cache_args or manual TeaCache configuration
- **Advanced Features**: Support for all cutting-edge WanVideo features
- **Error Handling**: Comprehensive error reporting and fallback outputs
- **Progress Tracking**: Detailed console output with feature usage reporting
- **Import Compatibility**: Robust handling of WanVideo package imports

**Usage:**
The TiledWan Video Sampler Simple node can be found in the "TiledWan" category in the ComfyUI node browser.

1. **Connect Required Inputs**: Connect your WanVideo model and image embeddings
2. **Configure Basic Parameters**: Set steps, CFG, shift, seed, and scheduler
3. **Enable Advanced Features**: 
   - Set `enable_teacache` to True and configure TeaCache parameters for speed optimization
   - Connect SLG arguments for sparse guidance
   - Add experimental arguments for cutting-edge features
4. **Optional Enhancements**: Connect text embeddings, input samples, or specialized embeddings
5. **Run the Workflow**: The node executes WanVideoSampler with all configured parameters

**Console Output includes:**
- **Parameter Summary**: List of all received parameters
- **Feature Detection**: Automatic detection of enabled advanced features (cache, SLG, experimental)
- **TeaCache Building**: Automatic TeaCache argument construction from manual inputs
- **Execution Progress**: Sampling progress and completion confirmation
- **Output Information**: Shape and details of generated latents
- **Error Handling**: Detailed error messages with full traceback

**Use Cases:**
- **Advanced Video Generation**: Access to all WanVideo capabilities in a single node
- **Feature Testing**: Experiment with cutting-edge WanVideo features
- **Performance Optimization**: Fine-tune caching and optimization settings
- **Research and Development**: Test new WanVideo innovations
- **Production Workflows**: Maximum control over video generation process

**Tips:**
- Start with basic required parameters and add advanced features incrementally
- Enable TeaCache with `enable_teacache=True` for faster inference
- Use pre-built argument objects when available for complex configurations
- Monitor console output to understand which features are being used
- The node handles missing optional arguments gracefully
- Consider VRAM usage when enabling multiple advanced features simultaneously