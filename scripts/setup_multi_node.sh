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

# Get WandB API key from environment, netrc file, or use default
if [ -z "$WANDB_API_KEY" ]; then
    # Try to get from existing wandb config
    WANDB_API_KEY=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])" 2>/dev/null || echo "")
fi
# Use default key if still not set
WANDB_API_KEY="${WANDB_API_KEY:-fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b}"

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

    # Pipe the remote setup body over stdin with `bash -s`, passing local values
    # as remote env vars. The heredoc delimiter is QUOTED, so none of the body is
    # expanded or mangled by the local shell — this avoids the single-quote
    # truncation bug that the old `ssh $node "bash -c '...'"` wrapper had.
    WORK_DIR="$WORK_DIR" WANDB_API_KEY="$WANDB_API_KEY" \
    RESOLVED_DATA_PATH="$RESOLVED_DATA_PATH" NODE_NAME="$node" \
    ssh $node "WORK_DIR='$WORK_DIR' WANDB_API_KEY='$WANDB_API_KEY' \
RESOLVED_DATA_PATH='$RESOLVED_DATA_PATH' NODE_NAME='$node' bash -s" <<'REMOTE_EOF'
        # These env vars are set on the remote command line above:
        #   WORK_DIR, WANDB_API_KEY, RESOLVED_DATA_PATH, NODE_NAME

        # Initialize conda
        if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
            source ~/miniconda3/etc/profile.d/conda.sh
        elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
            source ~/anaconda3/etc/profile.d/conda.sh
        elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
            source /opt/conda/etc/profile.d/conda.sh
        fi

        # Initialize conda for bash if not already done
        if ! grep -q "conda initialize" ~/.bashrc 2>/dev/null; then
            echo "Initializing conda for bash on $NODE_NAME..."
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
        if [ -n "$RESOLVED_DATA_PATH" ] && [ -d "$RESOLVED_DATA_PATH" ]; then
            mkdir -p /home/aiscuser/MiniOneRec/data 2>/dev/null || true
            if [ ! -e /home/aiscuser/MiniOneRec/data/GenRecDatasetV3 ]; then
                echo "Creating data symlink: /home/aiscuser/MiniOneRec/data/GenRecDatasetV3 -> $RESOLVED_DATA_PATH"
                ln -sf "$RESOLVED_DATA_PATH" /home/aiscuser/MiniOneRec/data/GenRecDatasetV3
            else
                echo "Data symlink already exists"
            fi
        fi

        # Check if MiniOneRec environment exists
        if conda env list | grep -q "^MiniOneRec "; then
            echo "Environment MiniOneRec already exists on $NODE_NAME"
        else
            echo "Creating MiniOneRec environment on $NODE_NAME..."
            conda create -n MiniOneRec python=3.11 -y
        fi

        # Activate environment
        conda activate MiniOneRec

        # Navigate to project directory
        cd $WORK_DIR

        # Define required versions for consistency across nodes
        TORCH_VERSION="2.6.0"
        TRANSFORMERS_VERSION="5.2.0"
        DEEPSPEED_VERSION="0.18.0"
        FLASH_ATTN_VERSION="2.7.3"
        TORCHREC_VERSION="0.8.0+cu124"
        FBGEMM_VERSION="0.8.0+cu124"

        # Check torch version and reinstall if different
        CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null | cut -d'+' -f1)
        if [[ "$CURRENT_TORCH" != "$TORCH_VERSION" ]]; then
            echo "Torch version mismatch on $NODE_NAME: $CURRENT_TORCH vs $TORCH_VERSION"
            echo "Installing torch==$TORCH_VERSION..."
            pip install -q torch==$TORCH_VERSION
        else
            echo "Torch version OK: $CURRENT_TORCH"
        fi

        # Fix torchvision compatibility (must match torch version)
        TORCHVISION_VERSION="0.21.0"
        CURRENT_TV=$(python -c "import torchvision; print(torchvision.__version__)" 2>/dev/null | cut -d'+' -f1)
        if [[ "$CURRENT_TV" != "$TORCHVISION_VERSION" ]]; then
            echo "Fixing torchvision on $NODE_NAME: $CURRENT_TV -> $TORCHVISION_VERSION"
            pip uninstall torchvision -y 2>/dev/null
            pip install -q torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cu124
        else
            echo "Torchvision version OK: $CURRENT_TV"
        fi

        # Fix NCCL version consistency (critical for multi-node training)
        NCCL_VERSION="2.21.5"
        CURRENT_NCCL=$(python -c "import nvidia.nccl; print(nvidia.nccl.__version__)" 2>/dev/null || echo "unknown")
        if [[ "$CURRENT_NCCL" != "$NCCL_VERSION" ]]; then
            echo "Fixing NCCL on $NODE_NAME: $CURRENT_NCCL -> $NCCL_VERSION"
            pip uninstall nvidia-nccl-cu11 nvidia-nccl-cu12 -y 2>/dev/null || true
            pip install -q nvidia-nccl-cu12==$NCCL_VERSION
        else
            echo "NCCL version OK: $CURRENT_NCCL"
        fi

        # Fix accelerate version consistency
        ACCELERATE_VERSION="1.10.1"
        CURRENT_ACC=$(python -c "import accelerate; print(accelerate.__version__)" 2>/dev/null || echo "unknown")
        if [[ "$CURRENT_ACC" != "$ACCELERATE_VERSION" ]]; then
            echo "Fixing accelerate on $NODE_NAME: $CURRENT_ACC -> $ACCELERATE_VERSION"
            pip install -q accelerate==$ACCELERATE_VERSION
        else
            echo "Accelerate version OK: $CURRENT_ACC"
        fi

        # SFT-launch-critical deps are installed EXPLICITLY and independently so a
        # weak/failing resolve of the heavy RL deps (verl git, vllm, ray) below can
        # never leave `import fire`/dataset/tokenizer missing -> ModuleNotFoundError
        # at trainer init. The mono `-r` resolve is intentionally BEST-EFFORT
        # (non-fatal): it may abort on a cold node image-cache, but that must not
        # block training.
        echo "Installing SFT-critical deps on $NODE_NAME..."
        # Each spec is QUOTED so bash never parses the >=/< as a shell
        # redirection. Unquoted `pyarrow<19` is read as "redirect stdin from
        # file 19" -> "19: No such file or directory" -> the whole pip line
        # fails -> fire never installs -> deepspeed import fire crash (r6).
        pip install -q --no-cache-dir 'fire==0.7.1' 'tqdm==4.67.1' 'safetensors==0.6.2' 'einops==0.8.0' 'datasets>=3.2.0' 'pyarrow<19'
        echo "Installing requirements on $NODE_NAME (best-effort)..."
        if [ -f requirements.txt ]; then
            pip install -q -r requirements.txt || echo "WARN: -r requirements.txt resolve failed on $NODE_NAME (non-fatal; SFT-critical deps already installed)"
        else
            echo "requirements.txt not found, installing core packages..."
            pip install -q transformers==$TRANSFORMERS_VERSION accelerate deepspeed==$DEEPSPEED_VERSION fire wandb scikit-learn tqdm
        fi

        # Enforce exact versions for core native deps (avoid mismatches across nodes)
        CURRENT_DS=$(python -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null || echo "unknown")
        if [[ "$CURRENT_DS" != "$DEEPSPEED_VERSION" ]]; then
            echo "Fixing DeepSpeed on $NODE_NAME: $CURRENT_DS -> $DEEPSPEED_VERSION"
            pip install -q deepspeed==$DEEPSPEED_VERSION
        else
            echo "DeepSpeed version OK: $CURRENT_DS"
        fi

        CURRENT_TR=$(python -c "import torchrec; print(torchrec.__version__)" 2>/dev/null || echo "unknown")
        if [[ "$CURRENT_TR" != "$TORCHREC_VERSION" ]]; then
            echo "Fixing torchrec on $NODE_NAME: $CURRENT_TR -> $TORCHREC_VERSION"
            pip install -q torchrec==$TORCHREC_VERSION --index-url https://download.pytorch.org/whl/cu124
        else
            echo "torchrec version OK: $CURRENT_TR"
        fi

        CURRENT_FB=$(python -c "import fbgemm_gpu; print(fbgemm_gpu.__version__)" 2>/dev/null || echo "unknown")
        if [[ "$CURRENT_FB" != "$FBGEMM_VERSION" ]]; then
            echo "Fixing fbgemm_gpu on $NODE_NAME: $CURRENT_FB -> $FBGEMM_VERSION"
            pip install -q fbgemm_gpu==$FBGEMM_VERSION --index-url https://download.pytorch.org/whl/cu124
        else
            echo "fbgemm_gpu version OK: $CURRENT_FB"
        fi

        # Flash Linear Attention (Qwen3.5 GatedDeltaNet hybrid backbone)
        # The transformers fast path gates on BOTH fla AND causal-conv1d being
        # importable. causal-conv1d has NO prebuilt wheel for torch2.6/cu124 (only
        # cu11 torch2.6 and cu13 torch>=2.7 wheels exist), so it tries a source
        # build that fails with a CUDA-mismatch error. Both installs below are
        # therefore BEST-EFFORT and NON-FATAL: if they fail we continue on the
        # slow torch path, so the eval always runs and delivers AUCs.
        if ! python -c "import fla" 2>/dev/null; then
            echo "Installing fla (Flash Linear Attention) on $NODE_NAME..."
            pip install -q flash-linear-attention==0.4.2 2>/dev/null && echo "fla installed on $NODE_NAME" || echo "WARN: fla install failed on $NODE_NAME; will use slow torch fallback"
        else
            echo "fla (Flash Linear Attention) OK on $NODE_NAME"
        fi
        # Best-effort causal-conv1d (may fail to build; not required).
        if ! python -c "import causal_conv1d" 2>/dev/null; then
            pip install -q "causal-conv1d>=1.4.0" 2>/dev/null && echo "causal-conv1d installed on $NODE_NAME" || echo "WARN: causal-conv1d not built on $NODE_NAME (ok, may fall back to slow path)"
        else
            echo "causal-conv1d OK on $NODE_NAME"
        fi

        # Check if flash-attn is installed (requires torch to be installed first)
        if python -c "import flash_attn" 2>/dev/null; then
            CURRENT_FA=$(python -c "import flash_attn; print(getattr(flash_attn, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
            if [[ "$CURRENT_FA" != "$FLASH_ATTN_VERSION" ]]; then
                echo "Fixing flash-attn on $NODE_NAME: $CURRENT_FA -> $FLASH_ATTN_VERSION"
                pip install -q flash-attn==$FLASH_ATTN_VERSION --no-build-isolation
            else
                echo "flash-attn version OK: $CURRENT_FA"
            fi
        else
            if python -c "import torch" 2>/dev/null; then
                echo "Installing flash-attn on $NODE_NAME..."
                pip install flash-attn==$FLASH_ATTN_VERSION --no-build-isolation
            else
                echo "Skipping flash-attn (torch not installed)"
            fi
        fi

        # Configure WandB authentication
        if [ -n "$WANDB_API_KEY" ]; then
            echo "Configuring WandB on $NODE_NAME..."
            python -c "import wandb; wandb.login(key=\"$WANDB_API_KEY\")" 2>/dev/null && echo "WandB configured successfully" || echo "WandB login failed"
        else
            echo "WANDB_API_KEY not set, skipping WandB login"
        fi

        echo "Setup complete on $NODE_NAME"
        echo "Python: $(which python)"
        echo "DeepSpeed version: $(python -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null || echo "Not installed")"
REMOTE_EOF
    if [ $? -ne 0 ]; then
        echo "Warning: Could not setup $node"
    fi

    echo ""
done

echo "================================"
echo "All nodes setup complete!"
echo "================================"
