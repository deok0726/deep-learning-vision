#!/bin/bash
echo "Experiments Start!"
cd /root/anomaly_detection
# class_label=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
class_label=(capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
for label in ${class_label[@]}; do
    python /root/anomaly_detection/run.py \
        --train_batch_size 4 \
        --test_batch_size 1 \
        --dataset_name MvTec \
        --dataset_root /hd/anomaly_detection/MVTec-AD/$label \
        --num_epoch 2000 \
        --train \
        --test \
        --learning_rate 0.1 \
        --learning_rate_decay \
        --learning_rate_decay_ratio 0.5 \
        --shuffle \
        --normalize \
        --resize \
        --resize_size 300 \
        --train_ratio 0.8 \
        --valid_ratio 0.2 \
        --test_tensorboard_shown_image_num 1 \
        --model_name ARNet \
        --exp_name MvTec_Data_ARNet_Model_target_$label
done
echo "Experiments Finished!"
