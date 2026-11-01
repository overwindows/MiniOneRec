import subprocess
import sys
from pathlib import Path

import pytest

from merge_mind import merge
from split_mind import parse_cuda_list, split


@pytest.mark.parametrize("value,expected", [
    (0, [0]), (3, [3]), ("0", [0]), (" 2,0 ", [2, 0]),
    ([2, 0], [2, 0]), ((0, 1), [0, 1]),
])
def test_gpu_ids_accept_fire_scalar_and_sequence_shapes(value, expected):
    assert parse_cuda_list(value) == expected


@pytest.mark.parametrize("value", [
    None, True, False, -1, 1.5, "", "0,", ",0", "0,bad", "0,-1",
    [], [0, 0], [0, True], [0, "1"], [0, -1],
])
def test_invalid_gpu_ids_are_not_silently_dropped(value):
    with pytest.raises(ValueError):
        parse_cuda_list(value)


def source(path, rows=4):
    path.write_text("".join(f"{i}\tU\ttime\t\tN1 N2\n" for i in range(1, rows + 1)),
                    encoding="utf-8")
    return path


def test_single_gpu_cli_round_trip_regression(tmp_path):
    behaviors = source(tmp_path / "behaviors.tsv")
    shards = tmp_path / "shards"
    split(str(behaviors), str(shards), 0)
    assert (shards / "0.tsv").read_bytes() == behaviors.read_bytes()
    (shards / "0.txt").write_text("".join(f"{i} N2 N1\n" for i in range(1, 5)), encoding="utf-8")
    output = tmp_path / "merged.txt"
    result = subprocess.run([
        sys.executable, "-X", "utf8", str(Path(__file__).with_name("merge_mind.py")),
        "--input_path", str(shards), "--output_path", str(output),
        "--cuda_list", "0", "--behaviors_path", str(behaviors), "--calculate_metrics", "false",
    ], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == (shards / "0.txt").read_text(encoding="utf-8")


def test_source_order_and_duplicate_candidate_exposures_are_preserved(tmp_path):
    behaviors = tmp_path / "behaviors.tsv"
    behaviors.write_text("1\tU\tt\t\tN1 N1 N2\n2\tU\tt\t\tN3-1 N4-0\n", encoding="utf-8")
    (tmp_path / "0.txt").write_text("2 N4 N3\n", encoding="utf-8")
    (tmp_path / "1.txt").write_text("1 N1 N2 N1\n", encoding="utf-8")
    output = tmp_path / "out.txt"
    merge(str(tmp_path), str(output), (0, 1), False, str(behaviors))
    assert output.read_text(encoding="utf-8") == "1 N1 N2 N1\n2 N4 N3\n"


def test_missing_shard_never_overwrites_an_existing_result(tmp_path):
    (tmp_path / "0.txt").write_text("1 N1 N2\n", encoding="utf-8")
    output = tmp_path / "old.txt"
    output.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        merge(str(tmp_path), str(output), [0, 1], False)
    assert output.read_text(encoding="utf-8") == "preserve\n"


@pytest.mark.parametrize("contents", ["", "\n", "1\n", "1 N1 N2\n1 N2 N1\n"])
def test_empty_malformed_and_duplicate_shards_fail_before_publication(tmp_path, contents):
    (tmp_path / "0.txt").write_text(contents, encoding="utf-8")
    output = tmp_path / "out.txt"
    with pytest.raises(ValueError):
        merge(str(tmp_path), str(output), 0, False)
    assert not output.exists()


@pytest.mark.parametrize("contents", [
    "1 N1\n2 N1 N2\n3 N1 N2\n4 N1 N2\n",
    "1 N1 N3\n2 N1 N2\n3 N1 N2\n4 N1 N2\n",
    "1 N1 N2\n2 N1 N2\n3 N1 N2\n",
    "1 N1 N2\n2 N1 N2\n3 N1 N2\n4 N1 N2\n5 N1 N2\n",
])
def test_exact_source_population_and_candidates_required(tmp_path, contents):
    behaviors = source(tmp_path / "behaviors.tsv")
    (tmp_path / "0.txt").write_text(contents, encoding="utf-8")
    output = tmp_path / "out.txt"
    with pytest.raises(ValueError):
        merge(str(tmp_path), str(output), 0, False, str(behaviors))
    assert not output.exists()


def test_explicit_per_shard_screening_limit_matches_evaluator(tmp_path):
    behaviors = source(tmp_path / "behaviors.tsv", rows=5)
    (tmp_path / "0.txt").write_text("1 N1 N2\n", encoding="utf-8")
    (tmp_path / "1.txt").write_text("3 N2 N1\n", encoding="utf-8")
    output = tmp_path / "out.txt"
    merge(str(tmp_path), str(output), [0, 1], False, str(behaviors), max_impressions=1)
    assert output.read_text(encoding="utf-8") == "1 N1 N2\n3 N2 N1\n"
    with pytest.raises(ValueError, match="exactly cover"):
        merge(str(tmp_path), None, [0, 1], False, str(behaviors))


def test_score_shards_ignore_rank_files_and_replay_complete_candidate_counts(tmp_path):
    behaviors = source(tmp_path / "behaviors.tsv", rows=2)
    (tmp_path / "0_scores.txt").write_text("1\t0.1\t-1\n2\t2\t3\n", encoding="utf-8")
    (tmp_path / "0.txt").write_text("not a score shard\n", encoding="utf-8")
    output = tmp_path / "scores.txt"
    merge(str(tmp_path), str(output), None, False, str(behaviors), score_files=True)
    assert output.read_text(encoding="utf-8") == "1\t0.1\t-1\n2\t2\t3\n"


@pytest.mark.parametrize("contents", ["1 nan 2\n", "1 inf 2\n", "1 bad 2\n", "1 1\n"])
def test_nonfinite_invalid_or_truncated_scores_fail(tmp_path, contents):
    behaviors = source(tmp_path / "behaviors.tsv", rows=1)
    (tmp_path / "0_scores.txt").write_text(contents, encoding="utf-8")
    output = tmp_path / "out.txt"
    with pytest.raises(ValueError):
        merge(str(tmp_path), str(output), 0, False, str(behaviors), score_files=True)
    assert not output.exists()
