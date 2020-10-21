#!/bin/bash
echo "Experiments Start!"
cd /root/anomaly_detection
# class_label=(0 1 2 3 4 5 6 7 8 9)
class_label=(3 4 5 6 7 8 9)
# class_label=(7 8 9)
for label in ${class_label[@]}; do
    python /root/anomaly_detection/run.py \
        --train_batch_size 32 \
        --test_batch_size 32 \
        --dataset_name MNIST \
        --dataset_root /hd \
        --channel_num 1 \
        --num_epoch 500 \
        --train \
        --test \
        --anomaly_ratio 0.9 \
        --target_label $label \
        --learning_rate 0.1 \
        --learning_rate_decay \
        --learning_rate_decay_ratio 0.5 \
        --shuffle \
        --normalize \
        --train_ratio 0.6 \
        --valid_ratio 0.3 \
        --test_ratio 0.1 \
        --test_tensorboard_shown_image_num 4 \
        --model_name ARNet \
        --exp_name MNIST_Data_ARNet_Model_target_SGD_nobilinear_lrdecay_$label
done
echo "Experiments Finished!"
