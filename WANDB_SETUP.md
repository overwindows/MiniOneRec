# Weights & Biases (wandb) Setup Guide

## Authentication Options

You have **three options** to authenticate with wandb:

### Option 1: Environment Variable (Recommended for Automated Training)

Set the `WANDB_API_KEY` environment variable before running training:

```bash
# Get your API key from: https://wandb.ai/authorize
export WANDB_API_KEY=your_api_key_here

# Then run training
bash sft.sh
```

**Or add it to your shell profile** (`~/.bashrc` or `~/.zshrc`):
```bash
echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

**Or set it in your training script** (`sft.sh`):
```bash
export WANDB_API_KEY=your_api_key_here
export NCCL_IB_DISABLE=1
PROCESS_NUM=4
# ... rest of script
```

### Option 2: Manual Login (One-time Setup)

Run once to authenticate:
```bash
wandb login
```

This will prompt you to enter your API key. After this, wandb will remember your credentials in `~/.netrc` or `~/.config/wandb/settings`.

**Note:** This requires interactive input, so it's not ideal for automated/headless training.

### Option 3: Config File

Create/edit `~/.netrc`:
```
machine api.wandb.ai
login user
password your_api_key_here
```

Or create `~/.config/wandb/settings`:
```ini
[default]
api_key = your_api_key_here
```

## How to Get Your API Key

1. Go to https://wandb.ai/authorize
2. Copy your API key
3. Use it in one of the methods above

## Current Codebase Behavior

Looking at the code:

- **SFT training** (`sft.py`): Uses `report_to=None` by default, but sets `WANDB_PROJECT` environment variable
- **RL training** (`rl.py`): Uses `report_to="wandb"` and sets `WANDB_MODE="offline"` (runs offline, syncs later)
- The code doesn't explicitly call `wandb.login()`, so it relies on:
  - Environment variables (`WANDB_API_KEY`)
  - Existing login credentials
  - Or will prompt if neither is available

## Recommended Setup for Automated Training

**Best approach:** Set the API key as an environment variable in your training script:

```bash
#!/bin/bash

# Set wandb API key (get from https://wandb.ai/authorize)
export WANDB_API_KEY=your_api_key_here

export NCCL_IB_DISABLE=1
PROCESS_NUM=4
MODEL_PATH=/nvmedata/hf_checkpoints/Qwen2.5-1.5B-Instruct

for category in "Industrial_and_Scientific"; do
    # ... rest of your script
    torchrun --nproc_per_node ${PROCESS_NUM} \
            sft.py \
            --wandb_project your_project_name \
            --wandb_run_name your_run_name \
            # ... other args
done
```

## Disable wandb (If Not Needed)

If you don't want to use wandb, you can disable it:

**Option 1:** Set environment variable:
```bash
export WANDB_MODE=disabled
bash sft.sh
```

**Option 2:** Modify the code to set `report_to=None` (already done in `sft.py`)

**Option 3:** Don't provide API key and wandb will run in offline mode (for RL training)

## Troubleshooting

### Issue: "wandb: ERROR Not logged in"

**Solution:** Set `WANDB_API_KEY` environment variable:
```bash
export WANDB_API_KEY=your_key
```

### Issue: "wandb: ERROR Network error"

**Solution:** Use offline mode (data syncs later):
```bash
export WANDB_MODE=offline
bash sft.sh
```

### Issue: Want to run without wandb

**Solution:** Disable it:
```bash
export WANDB_MODE=disabled
bash sft.sh
```

## Summary

✅ **For automated training:** Use `export WANDB_API_KEY=your_key` in your script  
✅ **For one-time setup:** Run `wandb login` once  
✅ **For no wandb:** Set `WANDB_MODE=disabled`

The codebase will automatically use the API key from the environment variable, so you don't need to modify any Python code!


