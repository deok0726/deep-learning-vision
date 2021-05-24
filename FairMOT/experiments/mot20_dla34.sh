cd src
python train.py mot --exp_id mot20_dla34 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 20 --lr_step '15' --data_cfg '../src/lib/cfg/mot20.json'
cd ..
