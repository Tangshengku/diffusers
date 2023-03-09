export MODEL_NAME="/home/dongk/dkgroup/tsk/projects/diffusers/ckpt/stable-diffusion-v1-5"
export TRAIN_DIR="/home/dongk/dkgroup/tsk/projects/data/coco/train"

accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
#   --max_train_steps=20000 \
  --num_train_epochs 5 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="fine_tuned_model" 