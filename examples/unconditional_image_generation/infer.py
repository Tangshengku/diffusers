import argparse, os

from diffusers import  DDPMPipeline, DDIMPipeline, UNet2DConditionModel, UNet2DModel
from transformers import CLIPTextModel
from diffusers import AutoencoderKL
from accelerate import Accelerator
from datasets import load_dataset
import torch
import time

torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/dongk/dkgroup/tsk/projects/diffusers/ckpt/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--samples_start",
        type=int,
        default=0,
        help="the start number of all samples, used to determine the name of saved images in multiple GPUs",
    )
    parser.add_argument(
        "--from-file",
        default="/home/dongk/dkgroup/tsk/projects/data/coco/annotations/val_30000.txt",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    accelerator = Accelerator()
    opt = parser.parse_args()
    model_path = "/home/dongk/dkgroup/tsk/projects/diffusers/ckpt/ddpm-cifar10-32"
    # sample_path = "/home/dongk/dkgroup/tsk/projects/data/coco/results/zero_shot_fine_tune_30000_step_fp32"
    base_count = opt.samples_start

    # Define Three component of Stable Diffusion
    unet = UNet2DModel.from_pretrained(
    model_path, torch_dtype=torch.float32)
    
    unet = accelerator.prepare(unet)
    
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     opt.pretrained_model_name_or_path, 
    #     unet=accelerator.unwrap_model(unet),
    #     torch_dtype=torch.float32
    #     )
    # model_id = "google/ddpm-cifar10-32"
    pipe = DDIMPipeline.from_pretrained(
        unet=accelerator.unwrap_model(unet),
        torch_dtype=torch.float32)
    pipe.to("cuda")
    start = time.time()
    num_image = 50000
    iters = num_image/opt.test_batch_size
    for iter in range(iters):
        images = pipe(batch_size=opt.test_batch_size).images
        accelerator.wait_for_everyone()
        images_all_process = accelerator.gather(images)
        for image in images_all_process:
            image = 1 / pipe.vae.config.scaling_factor * image.unsqueeze(0)
            image = pipe.vae.decode(image).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().detach().permute(0, 2, 3, 1).float().numpy()

            # image, _ = pipe.run_safety_checker(image, pipe._execution_device, torch.float32)
            images = pipe.numpy_to_pil(image)
            for img in images:
                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
            base_count += 1
    end = time.time()
    print(end - start)
if __name__ == "__main__":
    main()