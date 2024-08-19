#!/bin/bash
#first_noise_injection=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
first_noise_injection=(5 10 20 30 50 70 80 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
#bp_step_size=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
bp_step_size=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
for step_size in "${bp_step_size[@]}"; do
  for n1 in "${first_noise_injection[@]}"; do
    mpiexec --allow-run-as-root -n 1 python image_sample_superresolution.py --batch_size 1 --generator dummy --training_mode consistency_distillation \
 --model_path /opt/consistency_models/ckpts/cd_bedroom256_lpips.pt --attention_resolutions 32,16,8 --class_cond False \
 --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2  --resblock_updown True --use_fp16 True --weight_schedule uniform \
 --ts 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
   --steps 40 --out_dir /opt/consistency_models/samples/lsun_bedroom_256_sr --data_dir /opt/consistency_models/data_dir \
   --seed 1234 --wandb True --wandb_experiment_name "cm_bedroom_sr_0.05_n1_${n1}_bp_step_size_${step_size}" --bp_step_size $step_size \
   --n1 $n1
  done
done

#TODO:
# Run original first_noise_injection with low bp_step_size
# Rerun without update step (just denoising)
# Get images for CoSIGN, CM, naive upsampling and 1 NFE
