# LoRA Image Generation using Diffusers

This repository includes scripts for generating images using Stable Diffusion XL (SDXL) models with LoRA (Low-Rank Adaptation) weights. The scripts utilize the `diffusers` library and are tailored for SDXL 1.0 models. Please follow the instructions below for setting up and running the scripts.

## Files Description

- `test_model_loras.py`: A comprehensive script to batch test multiple SDXL 1.0 models with and without LoRA weights and log results to a CSV file.
- `single_lora_generation.py`: A focused script to generate images using the `dreamshaper-xl-lightning` model with detailed console logs.

Both scripts assume the use of CUDA-enabled GPUs for acceleration if available.

## Prerequisites

1. Python 3.x
2. Install the required packages:
   ```bash
   pip install torch diffusers requests
File Descriptions & Instructions
test_model_loras.py
This script tests multiple SDXL 1.0 models with and without LoRA weights. It logs the results into a CSV file (testing_results.csv). The script is designed to verify compatibility and performance across different SDXL 1.0 models.

Note
Compatibility: The LoRA weights are optimized for SDXL 1.0 models only. Check the script for a list of valid models.
Usage
python test_model_loras.py
Steps Logged:
Environment Setup:

Checks and ensures the setup of required directories and CUDA availability.
Downloads any necessary checkpoint files.
Disk Space Management:

Checks available disk space and performs cleanup if necessary.
Model Testing:

Iterates through specified models, generating images with and without LoRA weights.
Logs results, including any errors, to testing_results.csv.
Output
Images are saved in the generated_images directory, named as <model_name>_with_lora.png and <model_name>_without_lora.png.
Log file testing_results.csv with the details of each model test.
single_lora_generation.py
This script focuses on generating images using the dreamshaper-xl-lightning model with detailed and verbose console logs. It is intended for users who wish to see a clear step-by-step process and produce high-quality images.

Note
Model Selection: Uses dreamshaper-xl-lightning as the default model. For other valid SDXL 1.0 models, refer to test_model_loras.py.
Usage
python single_lora_generation.py
Steps Logged:
Environment Setup:

Ensures directory and CUDA setup.
Checks for and downloads necessary checkpoint files.
Disk Space Management:

Checks available disk space and performs cleanup if necessary.
Pipeline Loading:

Loads Stable Diffusion pipeline for the dreamshaper-xl-lightning model.
Image Generation:

Generates and saves images with and without LoRA weights.
Saves images in the generated_images directory, named as dreamshaper-xl-lightning_with_lora.png and dreamshaper-xl-lightning_without_lora.png.
Completion:

Provides cleanup and end-of-process console messages.
Conclusion
Both scripts are designed to ensure smooth and informed image generation using SDXL 1.0 models and LoRA weights. For best results, ensure your system has sufficient disk space and GPU acceleration is enabled.

Check out the generated images in the generated_images folder and review the testing_results.csv for detailed test logs when using test_model_loras.py.