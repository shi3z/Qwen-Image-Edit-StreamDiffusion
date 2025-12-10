#!/usr/bin/env python3
"""
Test script for Qwen-Image-Edit-2509
Step 2: Verify basic image editing functionality
"""

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import time
import os

def create_test_image(size=(512, 512)):
    """Create a simple test image"""
    import numpy as np
    # Create a simple gradient image with a circle
    img = np.zeros((*size, 3), dtype=np.uint8)

    # Blue gradient background
    for y in range(size[0]):
        for x in range(size[1]):
            img[y, x] = [
                int(100 + 100 * y / size[0]),  # R
                int(100 + 100 * x / size[1]),  # G
                150  # B
            ]

    # Draw a red circle in the center
    center = (size[0] // 2, size[1] // 2)
    radius = min(size) // 4
    for y in range(size[0]):
        for x in range(size[1]):
            if (y - center[0])**2 + (x - center[1])**2 < radius**2:
                img[y, x] = [200, 50, 50]  # Red circle

    return Image.fromarray(img)

def main():
    # Use GPU 7 which has most free memory
    GPU_ID = 7
    device = f'cuda:{GPU_ID}'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

    print("=" * 60)
    print("Qwen-Image-Edit-2509 Test Script")
    print("=" * 60)

    # Check GPU
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {GPU_ID} ({torch.cuda.get_device_name(0)})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load pipeline
    print("\n[1] Loading QwenImageEditPlusPipeline...")
    start_time = time.time()

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16
    )
    # Use CPU offloading to manage memory
    pipeline.enable_model_cpu_offload()

    load_time = time.time() - start_time
    print(f"    Pipeline loaded in {load_time:.1f}s")

    # Create test image
    print("\n[2] Creating test image...")
    test_img = create_test_image((512, 512))
    test_img.save("test_input.png")
    print("    Saved: test_input.png")

    # Run inference
    print("\n[3] Running image edit...")
    prompt = "Change the red circle to a blue star"

    start_time = time.time()

    inputs = {
        "image": [test_img],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 28,  # Reduced from 40 for faster test
        "guidance_scale": 1.0,
    }

    with torch.no_grad():
        output = pipeline(**inputs)

    inference_time = time.time() - start_time
    print(f"    Inference completed in {inference_time:.1f}s")
    print(f"    Steps: {inputs['num_inference_steps']}, Time per step: {inference_time/inputs['num_inference_steps']:.2f}s")

    # Save output
    print("\n[4] Saving output...")
    output_image = output.images[0]
    output_image.save("test_output.png")
    print("    Saved: test_output.png")

    # Memory stats
    if torch.cuda.is_available():
        print(f"\n[5] GPU Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    main()
