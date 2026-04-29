#!/usr/bin/env python3
import re
import sys


_KEYS = [
    "critic/score/mean",
    "val-aux/mind/reward/mean@1",
    "val-core/mind/acc/mean@1",
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/kl_loss",
    "prompt_length/mean",
    "response_length/mean",
]


def _parse_value(raw):
    cleaned = raw
    cleaned = cleaned.replace("np.float64(", "").replace("np.float32(", "")
    cleaned = cleaned.replace("np.int32(", "").replace("np.int64(", "")
    cleaned = cleaned.strip(")")
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _extract_metrics(line):
    metrics = {}
    for part in line.split(" - "):
        if ":" not in part:
            continue
        key, raw = part.split(":", 1)
        if key in _KEYS:
            metrics[key] = _parse_value(raw)
    return metrics


def _extract_step(line):
    match = re.search(r"step:(\d+)", line)
    return int(match.group(1)) if match else None


def _extract_val_metrics(line):
    metrics = {}
    for key in ("val-aux/mind/reward/mean@1", "val-core/mind/acc/mean@1"):
        match = re.search(rf"{re.escape(key)}':\s*([^,}}]+)", line)
        if match:
            metrics[key] = _parse_value(match.group(1).strip())
    return metrics


def main():
    in_traceback = False
    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        if not line:
            if in_traceback:
                print(line)
            continue

        # Detect start of a Python traceback
        if "Traceback (most recent call last)" in line:
            in_traceback = True
            print(line)
            continue

        # While inside a traceback, print every line until we hit the
        # exception line (non-indented line after the traceback frames)
        if in_traceback:
            print(line)
            # Exception lines are not indented and end the traceback
            if not line.startswith(" ") and not line.startswith("\t"):
                in_traceback = False
            continue

        if "Initial validation metrics:" in line:
            metrics = _extract_val_metrics(line)
            if metrics:
                summary = " | ".join(
                    f"{k}={_format_value(v)}" for k, v in metrics.items()
                )
                print(f"val_init | {summary}")
            continue

        if "step:" in line and " - " in line:
            step = _extract_step(line)
            metrics = _extract_metrics(line)
            if step is not None and metrics:
                ordered = [k for k in _KEYS if k in metrics]
                summary = " | ".join(
                    f"{k}={_format_value(metrics[k])}" for k in ordered
                )
                print(f"step {step} | {summary}")
            continue

        if (
            "WARNING" in line
            or "ERROR" in line
            or "Error" in line
            or "Exception" in line
            or "error" in line.lower() and line.lstrip().startswith(("omegaconf", "hydra", "Key", "Cannot", "config"))
        ):
            print(line)


if __name__ == "__main__":
    main()
