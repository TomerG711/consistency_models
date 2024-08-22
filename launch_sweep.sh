#!/bin/bash
#first_noise_injection=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
#first_noise_injection=(0.01)
first_noise_injection=(0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
denoiser_noise=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01 0.009 0.008 0.007 0.006 0.005 0.004 0.003 0.002 0.001)
bp_step_size=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
#bp_step_size=(1.0)
#TODO: Rerun bp <=0.3 with noise<0.01
#TODO: Full bp step (1.0)
#bp_step_size=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
for step_size in "${bp_step_size[@]}"; do
  for n1 in "${first_noise_injection[@]}"; do
    for dn in "${denoiser_noise[@]}"; do
    mpiexec --allow-run-as-root -n 1 python image_sample_superresolution.py --batch_size 1 --generator dummy --training_mode consistency_distillation \
 --model_path /opt/consistency_models/ckpts/cd_bedroom256_lpips.pt --attention_resolutions 32,16,8 --class_cond False \
 --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2  --resblock_updown True --use_fp16 True --weight_schedule uniform \
 --ts 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
   --steps 40 --out_dir /opt/consistency_models/samples/lsun_bedroom_256_sr --data_dir /opt/consistency_models/data_dir \
   --seed 1234 --wandb True --wandb_experiment_name "cm_bedroom_sr_0_n1_${n1}_bp_step_size_${step_size}_denoiser_noise_${dn}" --bp_step_size $step_size \
   --n1 $n1 --denoiser_noise $dn
    done
  done
done

#TODO:
# Run original first_noise_injection with low bp_step_size
# Rerun without update step (just denoising)
# Get images for CoSIGN, CM, naive upsampling and 1 NFE
