######################################################################################################
# Mind 2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_BERT/experiment.py
# - An execution file for multi-gpu training and hyperparameter searching.
#
# Version: 1.0
#######################################################################################################


import os, sys
sys.path.append('./')
import apex
import time

hyp=0 # 0 for no hyperparameter search
gpus = '0,1,2,3,4,5,6' # list for gpus
num_gpus = len(gpus.strip().split(','))
python_path = '~/.conda/envs/mind3/bin/python3.6' # path for python environment (conda)
os.system(f"{python_path} -m torch.distributed.launch --nproc_per_node={num_gpus} main.py --gpus {gpus} --hyp {hyp}")


# Below: hyper paramerter search
# hyp = 1 # 1 for hyperparameter tuning
# exp_num = 1 # numbering for experiments
# gpus = '0,1,2,3' # list for gpus
# num_gpus = len(gpus.strip().split(','))
# python_path = '~/.conda/envs/mind3/bin/python3.6' # path for python environment (conda)
# lr, d = 0.00016, 0.16
# # for lr in [0.0001, 0.0002]:
# #     for d in [0.16, 0.2, 0.24]:
# for wd in [256]:
#     os.system(f"{python_path} -m torch.distributed.launch --nproc_per_node={num_gpus} train.py --gpus {gpus} "
#               f"--hyp {hyp} --lr {lr} --dropout {d} --word_dim {wd}")
#     time.sleep(6000)
#     exp_num += 1
