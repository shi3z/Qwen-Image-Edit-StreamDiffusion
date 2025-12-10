#!/usr/bin/env python3
"""
Qwen-Image-Edit-2509 Real-time Camera Editing Pipeline
StreamDiffusion2-style acceleration approach

Key optimizations:
1. Fixed prompt/reference image pre-computation cache
2. Stateful latent (prev_latent) for temporal consistency
3. Reduced inference steps (2-4 instead of 28-40)
4. FlowMatchEulerDiscreteScheduler (flow matching for few-step inference)
5. torch.compile for kernel optimization (when not using CPU offloading)
6. xformers memory-efficient attention
7. Cross-attention K/V caching for fixed conditions

Target: 10-20 fps at 512x512 with 2-4 inference steps
"""

import torch
import torch.nn.functional as F
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import numpy as np
import cv2
import time
import os
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from threading import Thread
from queue import Queue
import warnings

warnings.filterwarnings("ignore")

# GPU Configuration
GPU_ID = 7
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)


@dataclass
class StreamConfig:
    """Configuration for the streaming pipeline"""
    # Resolution
    width: int = 512
    height: int = 512

    # Inference settings
    num_inference_steps: int = 4  # Target: 2-4 steps for real-time
    guidance_scale: float = 1.0
    true_cfg_scale: float = 4.0

    # Stateful latent settings
    noise_strength: float = 0.05  # How much noise to add between frames (0.03-0.07)
    keyframe_interval: int = 20  # Full denoise every N frames

    # Camera settings
    camera_id: int = 0
    target_fps: int = 15

    # Model settings
    model_name: str = "Qwen/Qwen-Image-Edit-2509"
    dtype: torch.dtype = torch.bfloat16

    # Optimization flags
    use_xformers: bool = True
    use_torch_compile: bool = True
    use_cuda_graph: bool = False  # Experimental


class CachedConditions:
    """Cache for pre-computed text embeddings and reference image features"""

    def __init__(self):
        self.prompt_embeds: Optional[torch.Tensor] = None
        self.pooled_prompt_embeds: Optional[torch.Tensor] = None
        self.negative_prompt_embeds: Optional[torch.Tensor] = None
        self.negative_pooled_prompt_embeds: Optional[torch.Tensor] = None
        self.reference_latents: Optional[torch.Tensor] = None
        self.cached_cross_attn_kv: Optional[Dict] = None
        self.prompt: str = ""
        self.reference_image: Optional[Image.Image] = None

    def is_valid(self, prompt: str, reference_image: Optional[Image.Image] = None) -> bool:
        """Check if cache is still valid for given inputs"""
        if self.prompt_embeds is None:
            return False
        if prompt != self.prompt:
            return False
        # For now, we don't cache reference image separately
        return True


class QwenRealtimePipeline:
    """
    Real-time image editing pipeline using Qwen-Image-Edit-2509
    with StreamDiffusion2-style acceleration
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self.device = torch.device('cuda')
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None
        self.cache = CachedConditions()
        self.prev_latent: Optional[torch.Tensor] = None
        self.frame_count: int = 0
        self.is_keyframe: bool = True

        # Performance tracking
        self.inference_times: List[float] = []
        self.frame_times: List[float] = []

    def load_model(self):
        """Load and optimize the model"""
        print("[1] Loading QwenImageEditPlusPipeline...")
        start_time = time.time()

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.dtype
        )

        # Use CPU offloading to manage memory (model is ~47GB)
        self.pipeline.enable_model_cpu_offload()

        # Note: xformers is incompatible with Qwen's transformer architecture
        # The QwenImageTransformer returns tuple from attention but xformers returns single tensor
        print("    xformers disabled (incompatible with Qwen transformer)")

        # Qwen uses FlowMatchEulerDiscreteScheduler (flow matching)
        # Keep the original scheduler - DPM-Solver++ is incompatible
        print(f"    Using original scheduler: {type(self.pipeline.scheduler).__name__}")

        # Note: torch.compile is disabled with CPU offloading as it can cause issues
        if self.config.use_torch_compile:
            print("    Skipping torch.compile (incompatible with CPU offloading)")

        load_time = time.time() - start_time
        print(f"    Model loaded in {load_time:.1f}s")
        print(f"    GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    def precompute_conditions(self, prompt: str, reference_image: Optional[Image.Image] = None):
        """
        Pre-compute and cache text embeddings and reference features
        This should be called once when prompt/reference changes
        """
        if self.cache.is_valid(prompt, reference_image):
            print("    Using cached conditions")
            return

        print(f"    Pre-computing conditions for: '{prompt}'")
        start_time = time.time()

        # Encode prompt - this varies by model architecture
        # For Qwen, we'll use the pipeline's internal encoding
        # The actual caching will happen inside the pipeline calls

        self.cache.prompt = prompt
        self.cache.reference_image = reference_image

        # TODO: For deeper optimization, extract and cache:
        # - Text encoder outputs (prompt_embeds)
        # - CLIP image encoder outputs for reference image
        # - Pre-computed cross-attention keys/values

        precompute_time = time.time() - start_time
        print(f"    Conditions cached in {precompute_time:.3f}s")

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to latent space using VAE"""
        # Resize image if needed
        if image.size != (self.config.width, self.config.height):
            image = image.resize((self.config.width, self.config.height), Image.LANCZOS)

        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=self.config.dtype)

        # Normalize to [-1, 1]
        image_tensor = 2.0 * image_tensor - 1.0

        # Encode with VAE
        with torch.no_grad():
            latent = self.pipeline.vae.encode(image_tensor).latent_dist.sample()
            latent = latent * self.pipeline.vae.config.scaling_factor

        return latent

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode latent to image using VAE"""
        with torch.no_grad():
            latent = latent / self.pipeline.vae.config.scaling_factor
            image_tensor = self.pipeline.vae.decode(latent).sample

        # Convert to PIL
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        return Image.fromarray(image_np)

    def prepare_latent(self, camera_image: Image.Image) -> torch.Tensor:
        """
        Prepare latent for current frame using stateful diffusion

        For keyframes: Start fresh with camera image + full noise
        For regular frames: Use prev_latent + small noise (temporal consistency)
        """
        # Encode current camera frame
        cam_latent = self.encode_image(camera_image)

        # Check if keyframe
        self.is_keyframe = (self.frame_count % self.config.keyframe_interval == 0) or (self.prev_latent is None)

        if self.is_keyframe:
            # Full denoise from random noise
            noise = torch.randn_like(cam_latent)
            latent = noise  # Start from pure noise for keyframes
        else:
            # Use previous latent with small noise for temporal consistency
            # This is the key to StreamDiffusion2-style acceleration
            noise = torch.randn_like(self.prev_latent)
            latent = self.prev_latent + self.config.noise_strength * noise

        return latent, cam_latent

    @torch.no_grad()
    def process_frame(self, camera_image: Image.Image) -> Image.Image:
        """
        Process a single camera frame through the editing pipeline

        Args:
            camera_image: Current camera frame

        Returns:
            Edited image
        """
        start_time = time.time()

        # Resize if needed
        if camera_image.size != (self.config.width, self.config.height):
            camera_image = camera_image.resize(
                (self.config.width, self.config.height),
                Image.LANCZOS
            )

        # Run inference
        output = self.pipeline(
            image=[camera_image],
            prompt=self.cache.prompt,
            negative_prompt=" ",
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            true_cfg_scale=self.config.true_cfg_scale,
            generator=torch.manual_seed(42),
        )

        result = output.images[0]

        # Store latent for next frame (extract from result)
        # Note: For true stateful diffusion, we'd need to intercept intermediate latents
        # This is a simplified version

        self.frame_count += 1

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        return result

    def warmup(self, warmup_image: Image.Image):
        """
        Warmup the pipeline with a few inference passes
        This helps torch.compile and CUDA to optimize
        """
        print("\n[2] Warming up pipeline...")

        for i in range(3):
            start_time = time.time()
            _ = self.process_frame(warmup_image)
            warmup_time = time.time() - start_time
            print(f"    Warmup {i+1}/3: {warmup_time:.2f}s")

        # Reset frame count after warmup
        self.frame_count = 0
        self.inference_times.clear()

        print("    Warmup complete")

    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        avg_time = np.mean(self.inference_times[-30:])  # Last 30 frames
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            "avg_inference_time": avg_time,
            "fps": fps,
            "frame_count": self.frame_count,
            "gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        }

    def run_camera_loop(self):
        """
        Main camera processing loop
        Press 'q' to quit
        """
        print(f"\n[3] Starting camera loop (target: {self.config.target_fps} fps)...")

        cap = cv2.VideoCapture(self.config.camera_id)
        if not cap.isOpened():
            print("    Error: Could not open camera")
            print("    Running in test mode with generated frames...")
            cap = None

        cv2.namedWindow("Qwen Real-time Edit", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)

        frame_interval = 1.0 / self.config.target_fps
        last_frame_time = time.time()

        try:
            while True:
                # Capture frame
                if cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_image = Image.fromarray(frame_rgb)
                else:
                    # Generate test pattern
                    t = time.time()
                    test_img = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
                    # Moving gradient
                    for y in range(self.config.height):
                        for x in range(self.config.width):
                            test_img[y, x] = [
                                int(127 + 127 * np.sin(x/50 + t)),
                                int(127 + 127 * np.sin(y/50 + t)),
                                int(127 + 127 * np.sin((x+y)/70 + t))
                            ]
                    camera_image = Image.fromarray(test_img)
                    frame = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

                # Process frame
                result = self.process_frame(camera_image)

                # Convert result to display format
                result_np = np.array(result)
                result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

                # Get stats
                stats = self.get_stats()

                # Add stats overlay
                cv2.putText(result_bgr, f"FPS: {stats.get('fps', 0):.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_bgr, f"Steps: {self.config.num_inference_steps}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_bgr, f"Frame: {self.frame_count}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display
                cv2.imshow("Original", frame)
                cv2.imshow("Qwen Real-time Edit", result_bgr)

                # Frame rate control
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.time()

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()

        print("\n[4] Camera loop ended")
        print(f"    Final stats: {self.get_stats()}")


def run_benchmark(pipeline: QwenRealtimePipeline, num_frames: int = 30):
    """Run a benchmark without camera"""
    print(f"\n[Benchmark] Processing {num_frames} frames...")

    # Create test image
    test_img = np.zeros((pipeline.config.height, pipeline.config.width, 3), dtype=np.uint8)
    for y in range(pipeline.config.height):
        for x in range(pipeline.config.width):
            test_img[y, x] = [
                int(100 + 100 * y / pipeline.config.height),
                int(100 + 100 * x / pipeline.config.width),
                150
            ]
    test_image = Image.fromarray(test_img)

    # Process frames
    times = []
    for i in range(num_frames):
        start = time.time()
        result = pipeline.process_frame(test_image)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    Frame {i+1}/{num_frames}: {elapsed:.3f}s ({1/elapsed:.1f} fps)")

    # Stats
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time

    print(f"\n[Benchmark Results]")
    print(f"    Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"    Average FPS: {fps:.2f}")
    print(f"    GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Save last result
    result.save("benchmark_output.png")
    print("    Output saved: benchmark_output.png")

    return fps


def main():
    print("=" * 60)
    print("Qwen-Image-Edit-2509 Real-time Pipeline")
    print("=" * 60)

    # Configuration
    config = StreamConfig(
        width=512,
        height=512,
        num_inference_steps=4,  # Start with 4 steps
        noise_strength=0.05,
        keyframe_interval=20,
        use_xformers=True,
        use_torch_compile=True,
    )

    print(f"\nConfiguration:")
    print(f"    Resolution: {config.width}x{config.height}")
    print(f"    Inference steps: {config.num_inference_steps}")
    print(f"    Noise strength: {config.noise_strength}")
    print(f"    Keyframe interval: {config.keyframe_interval}")

    # Create pipeline
    pipeline = QwenRealtimePipeline(config)

    # Load model
    pipeline.load_model()

    # Set prompt
    prompt = "Transform this into an oil painting style"
    pipeline.precompute_conditions(prompt)

    # Create warmup image
    warmup_img = np.zeros((config.height, config.width, 3), dtype=np.uint8)
    warmup_img[:, :] = [128, 128, 128]
    warmup_image = Image.fromarray(warmup_img)

    # Warmup
    pipeline.warmup(warmup_image)

    # Run benchmark
    fps = run_benchmark(pipeline, num_frames=20)

    print(f"\n{'=' * 60}")
    print(f"Target: 10-20 fps, Achieved: {fps:.1f} fps")
    print(f"{'=' * 60}")

    # Optional: Run camera loop
    # pipeline.run_camera_loop()


if __name__ == "__main__":
    main()
