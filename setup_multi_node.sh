#!/bin/bash

# Multi-node setup script for DeepSpeed training
# This script ensures all nodes have the proper environment

NODES=$(cat /job/hostfile | awk '{print $1}')
WORK_DIR="/scratch/azureml/cr/j/d12a2fad60394337a46dc66c5e8efc46/cap/data-capability/wd/INPUT_msndni/shares/users/wuc/MiniOneRec"

echo "Setting up environment on all nodes from /job/hostfile..."
cat /job/hostfile

for node in $NODES; do
    echo "================================"
    echo "Setting up $node..."
    echo "================================"

    ssh $node "bash -c '
        # Initialize conda
        if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
            source ~/miniconda3/etc/profile.d/conda.sh
        elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
            source ~/anaconda3/etc/profile.d/conda.sh
        elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
            source /opt/conda/etc/profile.d/conda.sh
        fi

        # Initialize conda for bash if not already done
        if ! grep -q \"conda initialize\" ~/.bashrc 2>/dev/null; then
            echo \"Initializing conda for bash on $node...\"
            if [ -f ~/miniconda3/bin/conda ]; then
                ~/miniconda3/bin/conda init bash
            elif [ -f ~/anaconda3/bin/conda ]; then
                ~/anaconda3/bin/conda init bash
            elif [ -f /opt/conda/bin/conda ]; then
                /opt/conda/bin/conda init bash
            fi
            source ~/.bashrc
        fi

        # Re-source conda after init
        if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
            source ~/miniconda3/etc/profile.d/conda.sh
        elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
            source ~/anaconda3/etc/profile.d/conda.sh
        elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
            source /opt/conda/etc/profile.d/conda.sh
        fi

        # Check if MiniOneRec environment exists
        if conda env list | grep -q \"^MiniOneRec \"; then
            echo \"Environment MiniOneRec already exists on $node\"
        else
            echo \"Creating MiniOneRec environment on $node...\"
            conda create -n MiniOneRec python=3.11 -y
        fi

        # Activate environment
        conda activate MiniOneRec

        # Navigate to project directory
        cd $WORK_DIR

        # Check if requirements are installed
        if python -c \"import transformers, deepspeed, torch\" 2>/dev/null; then
            echo \"Core packages already installed on $node\"
        else
            echo \"Installing requirements on $node...\"
            pip install -q -r requirements.txt
        fi

        # Check if flash-attn is installed
        if python -c \"import flash_attn\" 2>/dev/null; then
            echo \"flash-attn already installed on $node\"
        else
            echo \"Installing flash-attn on $node...\"
            pip install flash-attn --no-build-isolation
        fi

        echo \"Setup complete on $node\"
        echo \"Python: \$(which python)\"
        echo \"DeepSpeed version: \$(python -c \"import deepspeed; print(deepspeed.__version__)\" 2>/dev/null || echo \"Not installed\")\"
    '" || echo "Warning: Could not setup $node"

    echo ""
done

echo "================================"
echo "All nodes setup complete!"
echo "================================"
