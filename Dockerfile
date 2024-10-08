#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
##
##FROM nvcr.io/nvidia/pytorch:21.04-py3
#
#ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1
##
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libgl1-mesa-dev libopenmpi-dev git wget \
#    python3 python3-dev python3-pip python3-setuptools python3-wheel \
#    && apt-get clean && rm -rf /var/lib/apt/lists/*
##
#RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
#    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc
##
#RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel packaging mpi4py \
#    && pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
#    && pip3 install flash-attn==0.2.8 \
#    && pip3 install lmdb
##
###WORKDIR /home/
#RUN pip3 install -e git+https://github.com/openai/consistency_models.git@main#egg=consistency_models \
#    && ln -s /usr/bin/python3 /usr/bin/python
#
#RUN ln -s /usr/bin/python3 /usr/bin/python

#FOR SHORTER DOCKERFILE:
#FROM cm-base:latest
#FOR SHORTER TF DOCKERFILE:
FROM cm-tf-base:latest
#RUN apt-get -y update
#RUN apt-get -y install unzip

RUN pip3 install wandb
RUN pip3 install torchmetrics
RUN pip3 install lpips
#
#RUN pip3 install "blobfile>=1.0.5"
#RUN pip3 install scipy
#RUN pip3 install piq==0.7.0
#RUN pip3 install lmdb
#RUN pip3 install PyWavelets
COPY . /opt/consistency_models

WORKDIR /opt/consistency_models

RUN pip install matplotlib

# FOR RECREATING TF IMAGE
#RUN pip install -r /opt/consistency_models/evaluations/requirements.txt

#CMD python /opt/consistency_models/datasets/lsun_bedroom.py \
#--tomer_reader_images_path /opt/consistency_models/lsun/lsun_bedroom/data0 /opt/consistency_models/lsun/lsun_bedroom_processed
#CMD rm -rf /opt/consistency_models/lsun/lsun_bedroom

#CMD unzip /opt/consistency_models/lsun/archive.zip -d /opt/consistency_models/lsun/lsun_bedroom
#CMD sleep infinity

#CMD mpiexec --allow-run-as-root -n 1 python cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 \
#--scale_mode progressive --start_scales 2 --end_scales 150 --total_training_steps 1000000 --loss_norm lpips \
#--lr_anneal_steps 0 --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0  \
#--ema_rate 0.9999,0.99994,0.9999432189950708 --global_batch_size 8 --image_size 256 --lr 0.00005 --num_channels 256 \
#--num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform --use_fp16 True \
#--weight_decay 0.0 --weight_schedule uniform --data_dir /opt/consistency_models/lsun/lsun_bedroom_processed --log_interval 1 \
#--save_interval 20000 --wandb True --wandb_project consistency-models-256 --wandb_experiment_name CM_lsun_bedroom_256_batch_8_wavelets_dist_tv_1e-6 \
#--ckpts_dir /opt/consistency_models/ckpts/256_dist_tv_1e-6

#CMD sleep infinity
#
#CMD mpiexec --allow-run-as-root -n 1 python image_sample.py --batch_size 8 --generator determ-indiv --training_mode consistency_training \
#--sampler onestep --model_path /opt/consistency_models/ckpts/256_dist_tv_1e-5/target_model080000.pt --attention_resolutions 32,16,8 --class_cond False \
# --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 \
#  --num_res_blocks 2 --num_samples 50000 --resblock_updown True --use_fp16 True --weight_schedule uniform
#  --sampler multistep --ts 0,67,150 --steps 151


#CMD python convert_npz.py

#CMD python /opt/consistency_models/datasets/lsun_bedroom.py  /opt/consistency_models/lsun/full_bedroom_dataset/bedroom_train_lmdb/  /opt/consistency_models/lsun/full_bedroom_dataset/lsun_bedroom_processed


#CMD unzip -v /opt/consistency_models/lsun/full_bedroom_dataset/bedroom_train_lmdb.zip
#CMD sleep infinity

#CMD python /opt/consistency_models/evaluations/evaluator.py \
#/opt/consistency_models/samples/lsun_bedroom_256_ref_set/VIRTUAL_lsun_bedroom256.npz \
#/opt/consistency_models/samples/256_dist_tv_1e-5_ckpt_model80k_onestep/samples_50000x256x256x3.npz
#/opt/consistency_models/samples/64_lr_5e-5_wavelets_hh_100000_decrease_every_1000_ckpt_100k_onestep/samples_50000x64x64x3.npz

#CMD python /opt/consistency_models/convert_ref_batch_to_npz.py



CMD mpiexec --allow-run-as-root -n 1 python image_sample_superresolution.py --batch_size 1 --generator dummy --training_mode consistency_distillation \
 --model_path /opt/consistency_models/ckpts/cd_bedroom256_lpips.pt --attention_resolutions 32,16,8 --class_cond False \
 --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 \
  --num_res_blocks 2  --resblock_updown True --use_fp16 True --weight_schedule uniform \
 --ts 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40 \
# --ts 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
   --steps 40 --out_dir /opt/consistency_models/samples/lsun_bedroom_256_sr --data_dir /opt/consistency_models/data_dir \
   --seed 1234

#CMD bash launch_sweep.sh

#CMD sleep infinity