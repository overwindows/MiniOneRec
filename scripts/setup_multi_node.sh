#!/bin/bash

# Multi-node setup script for DeepSpeed training
# This script ensures all nodes have the proper environment

NODES=$(cat /job/hostfile | awk '{print $1}')
# Use current directory or specify WORK_DIR environment variable
WORK_DIR="${WORK_DIR:-$(pwd)}"

# Shared NFS path for DATA only (code is synced to each node)
SHARED_DATA_PATH="${SHARED_DATA_PATH:-/scratch/azureml/cr/j/*/cap/data-capability/wd/INPUT_msndni/shares/users/wuc/data/GenRecDatasetV3}"

# Resolve glob patterns to actual paths
RESOLVED_DATA_PATH=$(ls -d $SHARED_DATA_PATH 2>/dev/null | head -1)

# Check if the path is a shared path accessible from all nodes
# Prefer /home/aiscuser paths which are typically shared
if [[ "$WORK_DIR" == /scratch/* ]]; then
    echo "WARNING: $WORK_DIR may not be accessible on other nodes."
    echo "Consider using a shared path like /home/aiscuser/..."
fi

echo "Setting up environment on all nodes from /job/hostfile..."
echo "Work directory: $WORK_DIR"
cat /job/hostfile

# Get current node name
CURRENT_NODE=$(hostname)

for node in $NODES; do
    # Skip syncing to current node
    if [[ "$node" != "$CURRENT_NODE" ]]; then
        echo "================================"
        echo "Syncing code to $node..."
        echo "================================"
        # Create directory and sync code (excluding large files)
        ssh $node "mkdir -p $WORK_DIR" 2>/dev/null
        rsync -az --exclude='output_dir' --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' "$WORK_DIR/" "$node:$WORK_DIR/"
        echo "Code synced to $node"
    fi
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

        # Create symlink to shared NFS DATA path (code is rsynced separately)
        if [ -n \"$RESOLVED_DATA_PATH\" ] && [ -d \"$RESOLVED_DATA_PATH\" ]; then
            mkdir -p /home/aiscuser/MiniOneRec/data 2>/dev/null || true
            if [ ! -e /home/aiscuser/MiniOneRec/data/GenRecDatasetV3 ]; then
                echo \"Creating data symlink: /home/aiscuser/MiniOneRec/data/GenRecDatasetV3 -> $RESOLVED_DATA_PATH\"
                ln -sf \"$RESOLVED_DATA_PATH\" /home/aiscuser/MiniOneRec/data/GenRecDatasetV3
            else
                echo \"Data symlink already exists\"
            fi
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

        # Define required versions for consistency across nodes
        TORCH_VERSION=\"2.6.0\"
        TRANSFORMERS_VERSION=\"4.51.3\"
        DEEPSPEED_VERSION=\"0.18.0\"
        FLASH_ATTN_VERSION=\"2.7.3\"

        # Check torch version and reinstall if different
        CURRENT_TORCH=\$(python -c \"import torch; print(torch.__version__)\" 2>/dev/null | cut -d'+' -f1)
        if [[ \"\$CURRENT_TORCH\" != \"\$TORCH_VERSION\" ]]; then
            echo \"Torch version mismatch on $node: \$CURRENT_TORCH vs \$TORCH_VERSION\"
            echo \"Installing torch==\$TORCH_VERSION...\"
            pip install -q torch==\$TORCH_VERSION
        else
            echo \"Torch version OK: \$CURRENT_TORCH\"
        fi

        # Fix torchvision compatibility (must match torch version)
        TORCHVISION_VERSION=\"0.21.0\"
        CURRENT_TV=\$(python -c \"import torchvision; print(torchvision.__version__)\" 2>/dev/null | cut -d'+' -f1)
        if [[ \"\$CURRENT_TV\" != \"\$TORCHVISION_VERSION\" ]]; then
            echo \"Fixing torchvision on $node: \$CURRENT_TV -> \$TORCHVISION_VERSION\"
            pip uninstall torchvision -y 2>/dev/null
            pip install -q torchvision==\$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cu124
        else
            echo \"Torchvision version OK: \$CURRENT_TV\"
        fi

        # Fix NCCL version consistency (critical for multi-node training)
        NCCL_VERSION=\"2.21.5\"
        CURRENT_NCCL=\$(python -c \"import nvidia.nccl; print(nvidia.nccl.__version__)\" 2>/dev/null || echo \"unknown\")
        if [[ \"\$CURRENT_NCCL\" != \"\$NCCL_VERSION\" ]]; then
            echo \"Fixing NCCL on $node: \$CURRENT_NCCL -> \$NCCL_VERSION\"
            pip uninstall nvidia-nccl-cu11 nvidia-nccl-cu12 -y 2>/dev/null || true
            pip install -q nvidia-nccl-cu12==\$NCCL_VERSION
        else
            echo \"NCCL version OK: \$CURRENT_NCCL\"
        fi

        # Fix accelerate version consistency
        ACCELERATE_VERSION=\"1.10.1\"
        CURRENT_ACC=\$(python -c \"import accelerate; print(accelerate.__version__)\" 2>/dev/null || echo \"unknown\")
        if [[ \"\$CURRENT_ACC\" != \"\$ACCELERATE_VERSION\" ]]; then
            echo \"Fixing accelerate on $node: \$CURRENT_ACC -> \$ACCELERATE_VERSION\"
            pip install -q accelerate==\$ACCELERATE_VERSION
        else
            echo \"Accelerate version OK: \$CURRENT_ACC\"
        fi

        # Check if other requirements are installed
        if python -c \"import transformers, deepspeed\" 2>/dev/null; then
            echo \"Core packages already installed on $node\"
        else
            echo \"Installing requirements on $node...\"
            if [ -f requirements.txt ]; then
                pip install -q -r requirements.txt
            else
                echo \"requirements.txt not found, installing core packages...\"
                pip install -q transformers==\$TRANSFORMERS_VERSION accelerate deepspeed==\$DEEPSPEED_VERSION fire wandb scikit-learn tqdm
            fi
        fi

        # Check if flash-attn is installed (requires torch to be installed first)
        if python -c \"import flash_attn\" 2>/dev/null; then
            echo \"flash-attn already installed on $node\"
        else
            if python -c \"import torch\" 2>/dev/null; then
                echo \"Installing flash-attn on $node...\"
                pip install flash-attn==2.7.3 --no-build-isolation
            else
                echo \"Skipping flash-attn (torch not installed)\"
            fi
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
