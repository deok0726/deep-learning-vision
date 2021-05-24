cd src
python train.py mot --exp_id detrac_dla34 --load_model "../exp/mot/detrac_dla34/model_last.pth" --num_epochs 20 --lr_step '15' --data_cfg '../src/lib/cfg/detrac.json'
cd ..

