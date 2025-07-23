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
## WanVideoVACEpipe

**WanVideo VACE Pipeline** - A streamlined complete video generation pipeline that combines the core WanVideo nodes (VACEEncode → Sampler → Decode) in a single modular node. External arguments for TeaCache, SLG, and Experimental features are provided for maximum flexibility.

**Pipeline Flow:**
1. **VACE Encoding** → Converts input parameters to image embeddings
2. **WanVideo Sampling** → Generates video latents using all advanced features
3. **VAE Decoding** → Converts latents to final video output

**Required Inputs:**
- `model`: WANVIDEOMODEL - The WanVideo model
- `vae`: WANVAE - VAE model for encoding/decoding
- Standard sampler parameters: `steps`, `cfg`, `shift`, `seed`, `scheduler`

**VACE Encode Parameters:**
- `vace_width`: INT - Video width in pixels (default: 832)
- `vace_height`: INT - Video height in pixels (default: 480)
- `vace_num_frames`: INT - Number of frames to generate (default: 81)
- `vace_strength`: FLOAT - VACE encoding strength (default: 1.0)
- `vace_start_percent`: FLOAT - Start percentage for VACE application (default: 0.0)
- `vace_end_percent`: FLOAT - End percentage for VACE application (default: 1.0)

**Decode Parameters:**
- `decode_enable_vae_tiling`: BOOLEAN - Enable VAE tiling for memory efficiency (default: False)
- `decode_tile_x`: INT - VAE tile width in pixels (default: 272)
- `decode_tile_y`: INT - VAE tile height in pixels (default: 272)
- `decode_tile_stride_x`: INT - VAE tile stride X (default: 144)
- `decode_tile_stride_y`: INT - VAE tile stride Y (default: 128)

**External Arguments (Optional):**
- `cache_args`: CACHEARGS - Pre-built TeaCache arguments from WanVideoTeaCache node
- `slg_args`: SLGARGS - Pre-built SLG arguments from WanVideoSLG node  
- `experimental_args`: EXPERIMENTALARGS - Pre-built experimental arguments from WanVideoExperimentalArgs node

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

**Key Benefits:**
- **Modular Design**: Connect external WanVideoTeaCache, WanVideoSLG, and WanVideoExperimentalArgs nodes
- **Complete Pipeline**: Handles VACE encode → sampling → decode in one node
- **Memory Efficient**: Supports VAE tiling for large videos
- **Production Ready**: Comprehensive error handling and detailed logging
- **Maximum Flexibility**: Use external nodes to configure advanced features exactly as needed

**Usage Tip**: For maximum control, use separate WanVideoTeaCache, WanVideoSLG, and WanVideoExperimentalArgs nodes and connect their outputs to this pipeline node's `cache_args`, `slg_args`, and `experimental_args` inputs.
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
A complete and accurate wrapper for WanVideoSampler that provides 1:1 compatibility with the original node.

**Key Features:**
- **Exact Parameter Matching**: All parameters match the original WanVideoSampler exactly
- **Proper Defaults**: rope_function defaults to "comfy" (as in original)
- **Clean Interface**: TeaCache controlled exclusively through cache_args (no manual TeaCache inputs)
- **Deterministic Results**: Same seed produces same results as original node

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
- `rope_function`: COMBO - RoPE function type ("default", "comfy") - **Default: "comfy"**

**Optional Advanced Arguments:**
- `text_embeds`: WANVIDEOTEXTEMBEDS - Text embeddings for conditioning
- `samples`: LATENT - Input latent samples for img2img
- `feta_args`: FETAARGS - FETA optimization arguments
- `context_options`: CONTEXTOPTIONS - Context handling options
- `cache_args`: CACHEARGS - **Complete TeaCache control** (replaces all manual TeaCache inputs)
- `slg_args`: SLGARGS - SLG (Sparse Local Guidance) arguments
- `loop_args`: LOOPARGS - Loop generation arguments
- `experimental_args`: EXPERIMENTALARGS - Experimental features
- `sigmas`: SIGMAS - Custom noise schedule
- `unianimate_poses`: UNIANIMATE_POSES - UniAnimate pose data
- `fantasytalking_embeds`: FANTASYTALKING_EMBEDS - Fantasy talking embeddings
- `uni3c_embeds`: UNI3C_EMBEDS - Uni3C embeddings
- `multitalk_embeds`: MULTITALK_EMBEDS - MultiTalk embeddings
- `freeinit_args`: FREEINIT_ARGS - FreeInit arguments
- `teacache_args`: TEACACHE_ARGS - Alternative TeaCache arguments input

**Outputs:**
- `latents`: LATENT - Generated latent samples

**Important Changes from Previous Version:**
- ✅ **rope_function** now defaults to "comfy" (matches original behavior)
- ✅ **Removed manual TeaCache inputs** - use cache_args or teacache_args only
- ✅ **Cleaner interface** - no more confusing manual cache configuration
- ✅ **Deterministic results** - same parameters = same output as original

**Features:**
- **1:1 Compatibility**: Exact match with original WanVideoSampler behavior
- **Proper Parameter Defaults**: All defaults match the original node exactly
- **Comprehensive Coverage**: All WanVideoSampler arguments are exposed
- **Clean Architecture**: TeaCache controlled through proper argument objects only
- **Error Handling**: Robust error reporting with detailed traceback
- **Progress Tracking**: Detailed console output showing feature usage

**Usage:**
The TiledWan Video Sampler Simple node can be found in the "TiledWan" category in the ComfyUI node browser.

1. **Connect Required Inputs**: Connect your WanVideo model and image embeddings
2. **Configure Basic Parameters**: Set steps, CFG, shift, seed, and scheduler
3. **Use Proper rope_function**: Leave as "comfy" for standard behavior (default)
4. **Advanced Features**: 
   - Connect cache_args for TeaCache optimization (built from TeaCache nodes)
   - Connect slg_args for sparse guidance (built from SLG nodes)
   - Add experimental_args for cutting-edge features
5. **Run the Workflow**: Node executes with exact original behavior

**Console Output includes:**
- **Parameter Summary**: List of all received parameters
- **Feature Detection**: Shows which advanced features are enabled
- **Execution Progress**: Sampling progress and completion confirmation
- **Configuration Display**: Shows rope_function and other key settings
- **Output Information**: Shape and details of generated latents
- **Error Handling**: Detailed error messages with full traceback

**TeaCache Usage:**
- **Correct Method**: Connect a TeaCache node's output to cache_args input
- **Result**: Full TeaCache functionality with proper device management
- **No Manual Configuration**: All TeaCache settings handled by the TeaCache node

**Reproducibility:**
- **Same Seeds**: Identical results to original WanVideoSampler with same parameters
- **Deterministic**: No randomness introduced by the wrapper
- **Exact Behavior**: All parameter handling matches original implementation

**Use Cases:**
- **Drop-in Replacement**: Perfect substitute for original WanVideoSampler
- **Advanced Workflows**: Access all WanVideo features in single node
- **Debugging**: Compare outputs with original node using same parameters
- **Production**: Reliable, deterministic video generation
- **Research**: Access cutting-edge features while maintaining compatibility

**Tips:**
- Keep rope_function as "comfy" unless you specifically need "default"
- Use TeaCache nodes to create cache_args rather than trying to build manually
- Same seed + same parameters = identical output to original node
- Connect argument objects (cache_args, slg_args, etc.) rather than trying to configure manually
- Monitor console output to verify feature usage and configuration

### WanVideo VACE Pipeline
A complete, production-ready WanVideo pipeline that encapsulates multiple WanVideo components into a single, easy-to-use node. This is the ultimate WanVideo generation solution.

**Pipeline Components:**
- **WanVideoTeaCache** → cache_args → **WanVideoSampler**
- **WanVideoVACEEncode** → image_embeds → **WanVideoSampler** 
- **WanVideoExperimentalArgs** → experimental_args → **WanVideoSampler**
- **WanVideoSampler** → samples → **WanVideoDecode**
- **WanVideoDecode** → final video output

**Key Features:**
- **Complete Pipeline**: End-to-end video generation in a single node
- **Intelligent Connections**: Automatic parameter routing between internal components
- **Full Parameter Access**: All component parameters exposed except internal connections
- **Production Ready**: Optimized for real-world video generation workflows
- **Debug-Friendly**: Comprehensive logging for each pipeline stage

**Inputs:**
**Required Pipeline Inputs:**
- `model`: WANVIDEOMODEL - The WanVideo model for generation
- `vae`: WANVAE - VAE for encoding/decoding operations

**Core Sampling Parameters:**
- `steps`: INT - Number of sampling steps (default: 30)
- `cfg`: FLOAT - Classifier-free guidance scale (default: 6.0, range: 0-30)
- `shift`: FLOAT - Shift parameter for sampling (default: 5.0, range: 0-1000)
- `seed`: INT - Random seed for reproducibility (default: 0)
- `scheduler`: COMBO - Sampling scheduler (default: "unipc")

**VACE Encoding Parameters:**
- `vace_width`: INT - Video width (default: 832, step: 8)
- `vace_height`: INT - Video height (default: 480, step: 8)
- `vace_num_frames`: INT - Number of frames (default: 81, step: 4)
- `vace_strength`: FLOAT - VACE strength (default: 1.0, range: 0-10)
- `vace_start_percent`: FLOAT - VACE start percentage (default: 0.0)
- `vace_end_percent`: FLOAT - VACE end percentage (default: 1.0)

**TeaCache Parameters:**
- `teacache_rel_l1_thresh`: FLOAT - L1 threshold for cache (default: 0.3)
- `teacache_start_step`: INT - Cache start step (default: 1)
- `teacache_end_step`: INT - Cache end step (default: -1)
- `teacache_use_coefficients`: BOOLEAN - Use cache coefficients (default: True)
- `teacache_cache_device`: COMBO - Cache device ("main_device", "offload_device")
- `teacache_mode`: COMBO - Cache mode ("e", "e0")

**Experimental Features Parameters:**
- `exp_video_attention_split_steps`: STRING - Video attention split steps
- `exp_cfg_zero_star`: BOOLEAN - Enable CFG zero star
- `exp_use_zero_init`: BOOLEAN - Use zero initialization
- `exp_zero_star_steps`: INT - Zero star steps count
- `exp_use_fresca`: BOOLEAN - Enable Fresca optimization
- `exp_fresca_scale_low`: FLOAT - Fresca low scale (default: 1.0)
- `exp_fresca_scale_high`: FLOAT - Fresca high scale (default: 1.25)
- `exp_fresca_freq_cutoff`: INT - Fresca frequency cutoff (default: 20)

**Decode Parameters:**
- `decode_enable_vae_tiling`: BOOLEAN - Enable VAE tiling (default: False)
- `decode_tile_x`: INT - Tile width (default: 272)
- `decode_tile_y`: INT - Tile height (default: 272)  
- `decode_tile_stride_x`: INT - Tile stride X (default: 144)
- `decode_tile_stride_y`: INT - Tile stride Y (default: 128)

**Pipeline Control:**
- `enable_teacache`: BOOLEAN - Enable TeaCache optimization (default: True)
- `enable_experimental`: BOOLEAN - Enable experimental features (default: False)

**Optional Advanced Inputs:**
- `text_embeds`: WANVIDEOTEXTEMBEDS - Text conditioning
- `samples`: LATENT - Input latent samples for img2img
- `riflex_freq_index`: INT - Riflex frequency index
- `denoise_strength`: FLOAT - Denoising strength (default: 1.0)
- `force_offload`: BOOLEAN - Force model offloading (default: True)
- `batched_cfg`: BOOLEAN - Use batched CFG (default: False)
- `rope_function`: COMBO - RoPE function ("default", "comfy") - Default: "comfy"

**Optional Component Inputs:**
- `feta_args`: FETAARGS - FETA optimization arguments
- `context_options`: CONTEXTOPTIONS - Context handling options
- `loop_args`: LOOPARGS - Loop generation arguments
- `sigmas`: SIGMAS - Custom noise schedule
- `unianimate_poses`: UNIANIMATE_POSES - UniAnimate pose data
- `fantasytalking_embeds`: FANTASYTALKING_EMBEDS - Fantasy talking embeddings
- `uni3c_embeds`: UNI3C_EMBEDS - Uni3C embeddings
- `multitalk_embeds`: MULTITALK_EMBEDS - MultiTalk embeddings
- `freeinit_args`: FREEINIT_ARGS - FreeInit arguments

**External Overrides:**
- `external_slg_args`: SLGARGS - Pre-built SLG arguments (overrides internal SLG)

**VACE Optional Inputs:**
- `vace_input_frames`: IMAGE - Input frames for VACE
- `vace_ref_images`: IMAGE - Reference images for VACE
- `vace_input_masks`: MASK - Input masks for VACE
- `vace_tiled_vae`: BOOLEAN - Use tiled VAE in VACE

**Decode Optional Inputs:**
- `decode_normalization`: COMBO - Normalization method ("default", "minmax")

**Outputs:**
- `video`: IMAGE - Final generated video frames
- `latents`: LATENT - Generated latent samples (for further processing)

**Pipeline Flow:**
1. **TeaCache Setup** (if enabled): Creates cache arguments for acceleration
2. **Experimental Setup** (if enabled): Configures advanced experimental features
3. **VACE Encoding**: Processes input parameters into image embeddings
4. **WanVideo Sampling**: Generates latent samples using all configured optimizations
5. **Video Decoding**: Converts latents to final video frames

**Usage:**
The WanVideo VACE Pipeline node can be found in the "TiledWan" category in the ComfyUI node browser.

1. **Connect Core Inputs**: Connect your WanVideo model and VAE
2. **Configure Generation**: Set steps, CFG, shift, seed, and video dimensions
3. **Enable Optimizations**: 
   - Keep `enable_teacache=True` for faster generation
   - Set `enable_experimental=True` for cutting-edge features
4. **Fine-tune Components**: Adjust VACE, TeaCache, and decode parameters as needed
5. **Optional Enhancements**: Connect text embeddings, input frames, or external SLG args
6. **Run Pipeline**: Execute complete video generation in one step

**Console Output includes:**
- **Pipeline Overview**: Total parameters and enabled components
- **Import Status**: WanVideo package loading confirmation
- **Step-by-Step Progress**: Detailed progress for each pipeline stage
- **Component Configuration**: Settings for TeaCache, VACE, experimental features
- **Sampling Details**: Model, embeddings, and parameter information
- **Decode Information**: VAE settings and output characteristics
- **Error Handling**: Comprehensive error reporting with full traceback

**Advantages over Individual Nodes:**
- **Simplified Workflow**: Single node replaces 5+ individual nodes
- **Optimized Connections**: No manual parameter routing required
- **Reduced Complexity**: Fewer connections and potential error points
- **Better Performance**: Optimized internal data flow
- **Easier Debugging**: Centralized logging and error handling
- **Production Ready**: Designed for real-world video generation

**Use Cases:**
- **Production Video Generation**: Complete solution for high-quality video creation
- **Rapid Prototyping**: Quick testing of WanVideo capabilities
- **Batch Processing**: Efficient generation of multiple videos
- **Research and Development**: Easy experimentation with WanVideo features
- **Educational**: Learn complete WanVideo workflow in one node
- **Integration**: Simple integration into larger ComfyUI workflows

**Performance Tips:**
- Enable TeaCache for 2-3x speed improvement with minimal quality loss
- Use VAE tiling if running out of VRAM during decode
- Start with experimental features disabled, enable gradually
- Monitor console output for optimal parameter tuning
- Use appropriate video dimensions for your GPU memory
- Consider batch size vs. quality trade-offs

**Internal Connections (Handled Automatically):**
- TeaCache args → WanVideoSampler cache_args
- VACE embeds → WanVideoSampler image_embeds
- Experimental args → WanVideoSampler experimental_args
- Sampler output → Decode input samples
- All internal routing handled transparently

**Tips:**
- This node provides the complete WanVideo experience in a single, easy-to-use package
- All the power of the individual WanVideo nodes with simplified workflow
- Perfect for users who want complete WanVideo functionality without complex node routing
- Internal connections are optimized and error-proof
- Comprehensive debugging output helps troubleshoot any issues
- Designed for both beginners and advanced users