# Requirements
- python 3.6 or higher
- pytorch 1.1 or higher
- pytorch ignite 0.2.0 or higher
```bash
pip3 install requirements.txt
```

# Data Arguments
| dataname | data     | unimodal | n_epochs | n_layers | btl_size | target_class            |
| -------- | -------- | -------- | -------- | -------- | -------- | ----------------------- |
| STL      | steel    | FALSE    | 100      | 10       | 10       | 0,1,2,3,4,5,6           |
| OTTO     | otto     | FALSE    | 100      | 10       | 66       | 0,1,2,3,4,5,6,7,8       |
| SNSR     | sensor   | FALSE    | 100      | 10       | 17       | 1,2,3,4,5,6,7,8,9,10,11 |
| MNIST    | mnist    | FALSE    | 200      | 10       | 20       | 0,1,2,3,4,5,6,7,8,9     |
| F-MNIST  | fmnist   | FALSE    | 200      | 10       | 20       | 0,1,2,3,4,5,6,7,8,9     |
| MI-F     | cnc_mf   | TRUE     | 100      | 10       | 23       | 0                       |
| MI-V     | cnc_pvi  | TRUE     | 100      | 10       | 23       | 0                       |
| EOPT     | eo       | TRUE     | 100      | 10       | 6        | 0                       |
| NASA     | nasa     | TRUE     | 100      | 10       | 10       | 0                       |
| RARM     | robotarm | TRUE     | 100      | 10       | 3        | -1                      |
| STL      | steel    | TRUE     | 100      | 10       | 10       | 0,1,2,3,4,5,6           |
| OTTO     | otto     | TRUE     | 100      | 10       | 66       | 0,1,2,3,4,5,6,7,8       |
| SNSR     | sensor   | TRUE     | 100      | 10       | 17       | 1,2,3,4,5,6,7,8,9,10,11 |
| MNIST    | mnist    | TRUE     | 500      | 10       | 20       | 0,1,2,3,4,5,6,7,8,9     |
| F-MNIST  | fmnist   | TRUE     | 500      | 10       | 20       | 0,1,2,3,4,5,6,7,8,9     |

# Running experiments
## Single Experiments
### MNIST 
- To run multimodal
```bash
python novelty_detection.py --gpu_id 0 --n_epochs 200 --data mnist --target_class 1 --model ae --btl_size 20 --n_layers 10 --use_rapp --start_layer_index 1
```
- To run unimodal
```bash
python novelty_detection.py --gpu_id 0 --n_epochs 200 --data mnist --unimodal_normal --novelty_ratio 0.5 --target_class 1 --model ae --btl_size 20 --n_layers 10 --use_rapp --start_layer_index 1
```

### Faulty Steel 
- To run multimodal
```bash
python novelty_detection.py --gpu_id 0 --n_epochs 100 --data steel --target_class 1 --model ae --btl_size 20 --n_layers 10 --use_rapp --start_layer_index 0
```
- To run unimodal
```bash
python novelty_detection.py --gpu_id 0 --n_epochs 100 --data steel --unimodal_normal --novelty_ratio 0.5 --target_class 1 --model ae --btl_size 20 --n_layers 10 --use_rapp --start_layer_index 0
```

## Multiprocess Experiments
### MNIST
- To run multimodal
```bash
python repeat_novelty_detection.py --gpu_id 0 --n_epochs 200 --data mnist --target_class 0,1,2,3,4,5,6,7,8,9 --model ae,vae,aae --btl_size 20 --n_layers 10 --use_rapp --start_layer_index 1 --n_trials 1
```
- To run unimodal
```bash
python novelty_detection.py --gpu_id 0 --n_epochs 200 --data mnist --unimodal_normal --novelty_ratio 0.5 --target_class 0,1,2,3,4,5,6,7,8,9 --model ae,vae,aae --btl_size 20 --n_layers 10 --use_rapp --start_layer_index 0
```

### Faulty Steel 
- To run multimodal
```bash
python repeat_novelty_detection.py --gpu_id 0 --n_epochs 100 --data steel --target_class 0,1,2,3,4,5,6 --model ae,vae,aae --btl_size 10 --n_layers 10 --use_rapp --start_layer_index 0
```
- To run unimodal
```bash
python novelty_detection.py --gpu_id 0 --n_epochs 100 --data steel --unimodal_normal --novelty_ratio 0.5 --target_class 0,1,2,3,4,5,6 --model ae,vae,aae --btl_size 10 --n_layers 10 --use_rapp --start_layer_index 0
```

### Several GPU
- We also support multiprocessing with multi gpu
```bash
python repeat_novelty_detection.py --gpu_id 0,1 --n_epochs 100 --data steel --target_class 0,1,2,3,4,5,6 --model ae,vae,aae --btl_size 10 --n_layers 10 --use_rapp --start_layer_index 0
```

