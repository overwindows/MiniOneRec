#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline执行器 - CoT RL 训练版本

使用方法:
    python pipeline_executor.py --config config_mind_cot_rl.json --mount-dir /mnt/... --sft-model ... ...
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import signal
from datetime import datetime
from typing import Dict, Any

def setup_logging() -> logging.Logger:
    log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


class PipelineExecutor:

    def __init__(self, config_path: str, cli_variables: Dict[str, Any] = None, debug_mode: bool = False):
        self.config_path = config_path
        self.config = self._load_config()
        self.cli_variables = cli_variables or {}
        self.debug_mode = debug_mode
        self.code_dir = os.getcwd()
        self.variables = self._prepare_variables()
        self.current_working_dir = os.getcwd()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("=" * 80)
        logger.info(f"Pipeline: {self.config.get('pipeline_name', 'Unknown')}")
        logger.info(f"Description: {self.config.get('description', '')}")
        logger.info(f"Debug Mode: {self.debug_mode}")
        logger.info("=" * 80)

    def _signal_handler(self, signum, frame):
        logger.warning(f"接收到信号 {signum}，正在停止...")
        sys.exit(1)

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✓ 成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"✗ 加载配置文件失败: {e}")
            sys.exit(1)

    def _prepare_variables(self) -> Dict[str, Any]:
        variables = {}
        variables.update(self.config.get('base_paths', {}))
        variables.update(self.config.get('variables', {}))
        variables['code_dir'] = self.code_dir
        for key, value in self.cli_variables.items():
            if value is not None:
                variables[key] = value
                logger.info(f"变量: {key} = {value}")
        return variables

    def _replace_variables(self, text: str) -> str:
        result = text
        for key, value in self.variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def _execute_command(self, cmd_config: Dict[str, Any]) -> bool:
        name = cmd_config.get('name', 'Unnamed Command')
        description = cmd_config.get('description', '')
        command_template = cmd_config.get('command', '')
        working_dir = self._replace_variables(cmd_config.get('working_dir', self.current_working_dir))
        is_cd_command = cmd_config.get('is_cd_command', False)
        new_working_dir = cmd_config.get('new_working_dir', '')

        logger.info(f"\n{'=' * 80}")
        logger.info(f"执行: {name}")
        if description:
            logger.info(f"描述: {description}")
        logger.info(f"{'=' * 80}")

        if is_cd_command and new_working_dir:
            new_dir = self._replace_variables(new_working_dir)
            os.chdir(new_dir)
            self.current_working_dir = new_dir
            logger.info(f"✓ 切换工作目录到: {new_dir}")
            return True

        if working_dir and working_dir != self.current_working_dir:
            if os.path.exists(working_dir):
                os.chdir(working_dir)
                self.current_working_dir = working_dir
                logger.info(f"切换工作目录到: {working_dir}")
            else:
                logger.warning(f"工作目录不存在: {working_dir}")

        command = self._replace_variables(command_template)
        logger.info(f"命令: {command}")

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                cwd=working_dir if os.path.exists(working_dir) else None
            )

            while True:
                line = process.stdout.readline()
                if line:
                    logger.info(line.strip())
                elif process.poll() is not None:
                    break

            if process.returncode == 0:
                logger.info(f"✓ 命令执行成功")
                return True
            else:
                logger.error(f"✗ 命令执行失败，返回码: {process.returncode}")
                return False

        except Exception as e:
            logger.error(f"✗ 执行命令时出错: {e}")
            return False

    def run(self) -> bool:
        start_time = datetime.now()
        logger.info(f"\n{'=' * 80}")
        logger.info(f"开始执行Pipeline，开始时间: {start_time}")
        logger.info(f"{'=' * 80}")

        try:
            commands = self.config.get('commands', [])
            logger.info(f"共有 {len(commands)} 个命令需要执行")
            stop_on_error = not self.debug_mode

            for i, cmd_config in enumerate(commands, 1):
                is_debug_command = (
                    cmd_config.get('enabled', False) and
                    'sleep infinity' in cmd_config.get('command', '')
                )
                if is_debug_command and not self.debug_mode:
                    logger.info(f"--- 命令 {i}/{len(commands)} [跳过-非调试模式]: {cmd_config.get('name', '')} ---")
                    continue

                logger.info(f"\n--- 命令 {i}/{len(commands)} ---")
                if not self._execute_command(cmd_config):
                    logger.error(f"✗ 命令 {i} 失败")
                    if stop_on_error:
                        logger.error("终止Pipeline执行")
                        return False
                    else:
                        logger.warning("调试模式: 继续执行下一个命令")

            end_time = datetime.now()
            logger.info(f"\n{'=' * 80}")
            logger.info(f"✓ Pipeline执行成功! 总耗时: {end_time - start_time}")
            logger.info(f"{'=' * 80}")
            return True

        except Exception as e:
            import traceback
            logger.error(f"✗ Pipeline执行过程中出错: {e}")
            logger.error(traceback.format_exc())
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline执行器 - CoT RL 训练版本"
    )

    parser.add_argument("--config", required=True, help="配置文件路径（JSON格式）")

    # 基础参数
    parser.add_argument("--mount-dir", help="数据存储挂载路径")
    parser.add_argument("--sft-model", help="SFT checkpoint 路径")
    parser.add_argument("--data-root", help="数据集根目录（相对于挂载路径）")
    parser.add_argument("--output-root", help="输出目录根路径（相对于挂载路径）")

    # CoT RL 参数
    parser.add_argument("--cot-style", help="CoT 风格 (standard/category/detailed)")
    parser.add_argument("--reward-type", help="奖励类型")
    parser.add_argument("--max-response-length", help="最大响应长度")
    parser.add_argument("--max-prompt-length", help="最大提示长度")
    parser.add_argument("--train-batch-size", help="训练批次大小")
    parser.add_argument("--learning-rate", help="学习率")
    parser.add_argument("--kl-loss-coef", help="KL 损失系数")
    parser.add_argument("--total-epochs", help="训练轮数")
    parser.add_argument("--num-generations", help="每个样本的生成数量")
    parser.add_argument("--max-history", help="最大历史记录长度")
    parser.add_argument("--max-candidates", help="最大候选新闻数量")

    # 调试和评估参数
    parser.add_argument("--debug-mode", help="调试模式 (true/false)")
    parser.add_argument("--run-eval", help="训练后是否运行评估 (1/0)")
    parser.add_argument("--eval-split", help="评估数据集 (dev/test)")

    args = parser.parse_args()

    cli_variables = {}

    # 合成完整的 data_root 路径
    if args.mount_dir and args.data_root:
        cli_variables['data_root'] = f"{args.mount_dir}/{args.data_root}"
    elif args.data_root:
        cli_variables['data_root'] = args.data_root

    # 合成完整的 output_root 路径
    if args.mount_dir and args.output_root:
        cli_variables['output_root'] = f"{args.mount_dir}/{args.output_root}"
    elif args.output_root:
        cli_variables['output_root'] = args.output_root

    # SFT model 路径
    if args.sft_model:
        # 如果是相对路径，加上 mount_dir
        if args.mount_dir and not args.sft_model.startswith('/') and '/' in args.sft_model:
            cli_variables['sft_model'] = f"{args.mount_dir}/{args.sft_model}"
        else:
            cli_variables['sft_model'] = args.sft_model

    # CoT RL 参数
    if args.cot_style:
        cli_variables['cot_style'] = args.cot_style
    if args.reward_type:
        cli_variables['reward_type'] = args.reward_type
    if args.max_response_length:
        cli_variables['max_response_length'] = args.max_response_length
    if args.max_prompt_length:
        cli_variables['max_prompt_length'] = args.max_prompt_length
    if args.train_batch_size:
        cli_variables['train_batch_size'] = args.train_batch_size
    if args.learning_rate:
        cli_variables['learning_rate'] = args.learning_rate
    if args.kl_loss_coef:
        cli_variables['kl_loss_coef'] = args.kl_loss_coef
    if args.total_epochs:
        cli_variables['total_epochs'] = args.total_epochs
    if args.num_generations:
        cli_variables['num_generations'] = args.num_generations
    if args.max_history:
        cli_variables['max_history'] = args.max_history
    if args.max_candidates:
        cli_variables['max_candidates'] = args.max_candidates

    # 评估参数
    if args.run_eval:
        cli_variables['run_eval'] = args.run_eval
    if args.eval_split:
        cli_variables['eval_split'] = args.eval_split

    debug_mode = bool(args.debug_mode and args.debug_mode.lower() in ('true', '1', 'yes'))

    executor = PipelineExecutor(args.config, cli_variables, debug_mode=debug_mode)
    success = executor.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
