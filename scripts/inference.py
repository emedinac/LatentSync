# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper
from torch.profiler import profile, record_function, ProfilerActivity


def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available(
    ) and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

# torch.cuda.synchronize() # just left for the sake of documentation
# with profile(on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler/"),
#              record_shapes=True,
#              profile_memory=True,
#              with_stack=True,
#              with_flops=True,
#              activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#              ) as prof:

#     with record_function("Warmup"):  # test # just left for the sake of documentation
    # Warmup to avoid profiling the first few iterations
    # torch.randn(1, 3, 224, 224).to(dtype)
    # torch.cuda.synchronize()

# with record_function("DDIMScheduler"): # just left for the sake of documentation
    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError(
            "cross_attention_dim must be 768 or 384")

# with record_function("Audio2Feature"): # just left for the sake of documentation
    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

# with record_function("AutoencoderKL"): # just left for the sake of documentation
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=dtype).to("cuda")
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

# with record_function("UNet3DConditionModel"): # just left for the sake of documentation
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cuda",  # to cuda directly
    )  # .to(dtype=dtype)
    unet = unet.to(dtype=dtype)

# with record_function("UNet3DConditionModel"): # just left for the sake of documentation
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda").to(dtype=dtype)

# with record_function("DeepCacheSDHelper"): # just left for the sake of documentation

    # use DeepCache
    if args.enable_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

# with record_function("pipeline"): # just left for the sake of documentation
    print(f"Initial seed: {torch.initial_seed()}")
    with torch.inference_mode():
        pipeline(
            video_path=args.video_path,
            audio_path=args.audio_path,
            video_out_path=args.video_out_path,
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir=args.temp_dir,
        )

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    # prof.export_chrome_trace("pipeline_trace.json")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str,
                        default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)
