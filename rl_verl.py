from fire import Fire

from minionerec_verl_trainer import train_verl


def main(
    model_path: str = "",
    train_parquet: str = "",
    eval_parquet: str = "",
    output_dir: str = "output_dir/verl_rl",
    reward_type: str = "rule",
    num_generations: int = 16,
    train_batch_size: int = 1024,
    max_prompt_length: int = 512,
    max_response_length: int = 128,
    learning_rate: float = 1e-6,
    total_epochs: int = 1,
    temperature: float = 1.0,
    rollout_name: str = "hf",
    ppo_mini_batch_size: int = 256,
    ppo_micro_batch_size_per_gpu: int = 32,
    kl_loss_coef: float = 0.001,
    kl_loss_type: str = "low_var_kl",
    wandb_project: str = "",
    wandb_run_name: str = "",
    save_freq: int = 20,
    test_freq: int = 5,
    nnodes: int = 1,
    n_gpus_per_node: int = 8,
    sid_info_file: str = "",
    ada_path: str = "",
    cf_path: str = "",
    sasrec_len_seq: int = 10,
):
    train_verl(
        model_path=model_path,
        train_parquet=train_parquet,
        eval_parquet=eval_parquet,
        output_dir=output_dir,
        reward_type=reward_type,
        num_generations=num_generations,
        train_batch_size=train_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        learning_rate=learning_rate,
        total_epochs=total_epochs,
        temperature=temperature,
        rollout_name=rollout_name,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=ppo_micro_batch_size_per_gpu,
        kl_loss_coef=kl_loss_coef,
        kl_loss_type=kl_loss_type,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        save_freq=save_freq,
        test_freq=test_freq,
        nnodes=nnodes,
        n_gpus_per_node=n_gpus_per_node,
        sid_info_file=sid_info_file,
        ada_path=ada_path,
        cf_path=cf_path,
        sasrec_len_seq=sasrec_len_seq,
    )


if __name__ == "__main__":
    Fire(main)
