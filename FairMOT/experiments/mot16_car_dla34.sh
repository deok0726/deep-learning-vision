cd src
python train.py mot --exp_id mot16_car_dla34 --gpus 3,4 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mot16_car.json'
cd ..


