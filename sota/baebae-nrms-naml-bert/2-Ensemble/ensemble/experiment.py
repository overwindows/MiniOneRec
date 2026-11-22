######################################################################################################
# Mind 2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/ensemble/experiment.py
# - Sequentially executing the main file by following a defined setting.
#
# Version: 1.0
#######################################################################################################

import itertools
import os
from multiprocessing import Process
from utils import _process_command_list_subprocess

##################### Define Settings ############################

model_combs = [] # split by comma (,) e.g. "MODEL-A,MODEL-B,MODEL-C"
weight_combs = [] # split by comman (,) e.g. "0.7,0.3"
# '999' means equall weights

# Define models and weights
model_list = ['NB_exp1_14','NB_exp4_4','NB_exp2_6','NB_exp5_7','N_6951','N_6982']
weight_combs.append('999')
all_combs = list(itertools.product([0,1], repeat=len(model_list)))
for comb in all_combs:
    new_comb = [model_list[i] for i in range(len(model_list)) if comb[i] == 1]
    if len(new_comb) == 0:
        continue
    model_combs.append(','.join(new_comb))
    weight_combs.append('999')

merge_types = ['score']  # score, rank, z_score,
num_workers = 1
dev_test = ['dev'] # dev or test
out_path = '../out/final_ensemble'
exp_num = 1
# Increase exp_num if foler exists
while os.path.isdir(out_path + f'{exp_num}'):
    exp_num +=1
out_path += f'/{exp_num}'

os.makedirs(out_path, exist_ok=True)

ERROR_LOG_FILE = out_path + '/error_log.txt'
if os.path.isfile(ERROR_LOG_FILE):
    os.remove(ERROR_LOG_FILE)
##################################################################


# Process
def do_command(cmd):
    os.system(cmd)


# Make list of commands
cmd_list = []
idx=0
for model_comb, weight_comb in zip(model_combs, weight_combs):
    if len(model_comb) == 0:
        continue
    for dt in dev_test:
        for merge_type in merge_types: 
            cmd = f'python main.py --model_comb {model_comb} --dev_test {dt} --out_path {out_path}'+\
                  f' --merge_type {merge_type} --idx {idx} --weight_comb {weight_comb}'
            cmd_list.append(cmd)
            idx+=1
            
print("Number of commands: ", len(cmd_list))

cmds = [ [] for _ in range(num_workers)]
for i in range(len(cmd_list)):
    cmds[i%num_workers].append(cmd_list[i])

procs = []
for wid in range(num_workers):
    new_proc = Process(target=_process_command_list_subprocess,
                        args=(cmds[wid], ERROR_LOG_FILE))
    procs.append(new_proc)
    new_proc.start()

for proc in procs:
    proc.join()

