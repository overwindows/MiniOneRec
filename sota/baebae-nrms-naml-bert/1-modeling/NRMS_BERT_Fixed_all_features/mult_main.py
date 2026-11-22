import os
import apex
import time

hyp=0 # No hyperparameter tuning
gpus = '0,1,2,3'
num_gpus = len(gpus.strip().split(','))
python_path = '~/.conda/envs/mind3/bin/python3.6'
os.system(f"{python_path} -m torch.distributed.launch --nproc_per_node={num_gpus} main.py --gpus {gpus} --hyp {hyp}")


# Below: hyper paramerter search
# exp_num = 1
# gpus = '0,1,2,3'
# num_gpus = len(gpus.strip().split(','))
# python_path = '~/.conda/envs/mind3/bin/python3.6'
# hyp = 1
# lr, d = 0.00016, 0.16
# # for lr in [0.0001, 0.0002]:
# #     for d in [0.16, 0.2, 0.24]:
# for wd in [256]:
#     os.system(f"{python_path} -m torch.distributed.launch --nproc_per_node={num_gpus} train.py --gpus {gpus} "
#               f"--hyp {hyp} --lr {lr} --dropout {d} --word_dim {wd}")
#     time.sleep(6000)
#     exp_num += 1
