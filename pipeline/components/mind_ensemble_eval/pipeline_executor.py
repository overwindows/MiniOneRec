#!/usr/bin/env python3
"""
Pipeline executor for MIND ensemble evaluation.
Driven by config_mind_ensemble_eval.json.
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
        logger.warning(f"Signal {signum} received, stopping...")
        sys.exit(1)

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✓ Loaded config: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"✗ Failed to load config: {e}")
            sys.exit(1)

    def _prepare_variables(self) -> Dict[str, Any]:
        variables = {}
        variables.update(self.config.get('base_paths', {}))
        variables.update(self.config.get('variables', {}))
        variables['code_dir'] = self.code_dir
        for key, value in self.cli_variables.items():
            if value is not None:
                variables[key] = value
                logger.info(f"Variable: {key} = {value}")
        return variables

    def _replace_variables(self, text: str) -> str:
        result = text
        for key, value in self.variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def _execute_command(self, cmd_config: Dict[str, Any]) -> bool:
        name = cmd_config.get('name', 'Unnamed')
        description = cmd_config.get('description', '')
        command_template = cmd_config.get('command', '')
        working_dir = self._replace_variables(cmd_config.get('working_dir', self.current_working_dir))
        is_cd_command = cmd_config.get('is_cd_command', False)
        new_working_dir = cmd_config.get('new_working_dir', '')

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Step: {name}")
        if description:
            logger.info(f"Description: {description}")
        logger.info(f"{'=' * 80}")

        if is_cd_command and new_working_dir:
            new_dir = self._replace_variables(new_working_dir)
            os.chdir(new_dir)
            self.current_working_dir = new_dir
            logger.info(f"✓ Changed directory to: {new_dir}")
            return True

        if working_dir and working_dir != self.current_working_dir:
            if os.path.exists(working_dir):
                os.chdir(working_dir)
                self.current_working_dir = working_dir
                logger.info(f"Changed directory to: {working_dir}")

        command = self._replace_variables(command_template)
        logger.info(f"Command: {command}")

        try:
            process = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', bufsize=1,
                cwd=working_dir if os.path.exists(working_dir) else None
            )
            while True:
                line = process.stdout.readline()
                if line:
                    logger.info(line.strip())
                elif process.poll() is not None:
                    break
            if process.returncode == 0:
                logger.info(f"✓ Step succeeded")
                return True
            else:
                logger.error(f"✗ Step failed (exit code {process.returncode})")
                return False
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            return False

    def run(self) -> bool:
        start_time = datetime.now()
        commands = self.config.get('commands', [])
        logger.info(f"\nStarting pipeline with {len(commands)} steps at {start_time}")
        stop_on_error = not self.debug_mode

        for i, cmd_config in enumerate(commands, 1):
            is_debug_command = (
                cmd_config.get('enabled', False) and
                'sleep infinity' in cmd_config.get('command', '')
            )
            if is_debug_command and not self.debug_mode:
                logger.info(f"--- Step {i}/{len(commands)} [skipped - not debug mode]: {cmd_config.get('name', '')} ---")
                continue

            logger.info(f"\n--- Step {i}/{len(commands)} ---")
            if not self._execute_command(cmd_config):
                logger.error(f"✗ Step {i} failed")
                if stop_on_error:
                    logger.error("Stopping pipeline")
                    return False
                else:
                    logger.warning("Debug mode: continuing to next step")

        end_time = datetime.now()
        logger.info(f"\n✓ Pipeline completed! Total time: {end_time - start_time}")
        return True


def main():
    parser = argparse.ArgumentParser(description="MIND Ensemble Eval Pipeline Executor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mount-dir", required=True)
    parser.add_argument("--wave1-models", required=True, help="Comma-separated Wave 1 model paths (relative to mount)")
    parser.add_argument("--wave2-models", default="", help="Comma-separated Wave 2 model paths (relative to mount, optional)")
    parser.add_argument("--data-root", default="shares/users/wuc/data/MIND_large")
    parser.add_argument("--output-root", default="shares/users/wuc/output_dir")
    parser.add_argument("--output-subdir", default="ensemble_results")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--mind-size", default="large")
    parser.add_argument("--max-history", default="30")
    parser.add_argument("--batch-size", default="8")
    parser.add_argument("--run-convert", default="0")
    parser.add_argument("--debug-mode", default="false")
    args = parser.parse_args()

    mount = args.mount_dir.rstrip("/")

    # Build full model path strings (space-separated for bash script)
    def expand_paths(csv: str) -> str:
        if not csv.strip():
            return ""
        return " ".join(f"{mount}/{p.strip()}" for p in csv.split(",") if p.strip())

    wave1_paths = expand_paths(args.wave1_models)
    wave2_paths = expand_paths(args.wave2_models)

    cli_variables = {
        'wave1_model_paths': wave1_paths,
        'wave2_model_paths': wave2_paths,
        'data_root': f"{mount}/{args.data_root}",
        'output_root': f"{mount}/{args.output_root}",
        'output_subdir': args.output_subdir,
        'split': args.split,
        'mind_size': args.mind_size,
        'max_history': args.max_history,
        'batch_size': args.batch_size,
        'run_convert': args.run_convert,
    }

    debug_mode = args.debug_mode.lower() in ('true', '1', 'yes')
    executor = PipelineExecutor(args.config, cli_variables, debug_mode=debug_mode)
    sys.exit(0 if executor.run() else 1)


if __name__ == "__main__":
    main()
