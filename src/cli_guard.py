"""Fail fast on CLI flags the trainer does not accept.

WHY THIS EXISTS
---------------
`fire.Fire(train)` does NOT reject unknown flags before running. Measured with
fire 0.7.1 (see /tmp/firetest3.py):

    $ python trainer.py --a 5 --use_dedup_backfill True --b 7
    [t+0.000] TRAIN ENTERED a=5 b=7      <- train() ran with the flag DROPPED
    [t+3.000] TRAIN RETURNED
    ERROR: Could not consume arg: --use_dedup_backfill
    $ echo $?
    2

So an unknown flag trains to completion and only then exits 2. In a Slurm log
that error is invisible between thousands of training lines -- and the run
silently trains with the DEFAULT value of whatever the flag was supposed to
change. That is exactly how the three 9B "lg_dedupbf" runs ended up training
`history_selection='recent'` under a dedup_backfill label: they were
duplicates of lg_9b, not the experiment they claimed to be.

`check_unknown_flags` turns that into an immediate, loud failure BEFORE any
GPU work starts.

USAGE
    from cli_guard import check_unknown_flags
    if __name__ == "__main__":
        check_unknown_flags(train, argv=sys.argv[1:])
        fire.Fire(train)
"""
import inspect
import sys

# DeepSpeed's launcher appends --local_rank=<n> to every rank's argv. It is
# a real parameter of every trainer in this repo, but the launcher also passes
# it to the script itself, so it must be tolerated explicitly.
LAUNCHER_INJECTED = {"local_rank"}


def _accepted_params(fn):
    """Every parameter name `fn` accepts, including **kwargs targets."""
    sig = inspect.signature(fn)
    names = set(sig.parameters)
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return None  # accepts anything -> nothing to check
    return names


def check_unknown_flags(fn, argv=None):
    """Raise SystemExit if `argv` contains a flag `fn` does not accept.

    Only `--flag value` and `--flag=value` forms are inspected; bare
    positional args are left to fire.
    """
    if argv is None:
        argv = sys.argv[1:]
    accepted = _accepted_params(fn)
    if accepted is None:
        return
    accepted |= LAUNCHER_INJECTED

    unknown = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            name = a[2:].split("=", 1)[0]
            if name and name not in accepted:
                unknown.append(a)
            # Skip the value of a "--flag value" pair so the value is not
            # itself mistaken for a flag.
            if "=" not in a and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                i += 1
        i += 1

    if unknown:
        print("=" * 70, file=sys.stderr)
        print("FATAL: unknown CLI flag(s) for "
              f"{getattr(fn, '__name__', fn)}:", file=sys.stderr)
        for u in unknown:
            print(f"  {u}", file=sys.stderr)
        print(file=sys.stderr)
        print("These flags would be SILENTLY DROPPED and the run would train "
              "with the default value instead.", file=sys.stderr)
        print("Accepted flags:", ", ".join(sorted(accepted)), file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        raise SystemExit(2)
