from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from spectral_metrics import (
    conceptor_eigenvalues,
    evaluate_finite_formula,
    hedges_g,
    is_complete_ordering,
    mann_whitney_auto_method,
    oriented_two_sided_mann_whitney,
    participation_ratio_effective_rank,
    shannon_effective_rank,
    spectral_variance_edge_fraction,
    spectrum_summary,
    validate_run_binding,
)


def test_conceptor_eigenvalues_matches_dense_reference() -> None:
    generator = torch.Generator().manual_seed(17)
    activations = torch.randn(7, 11, generator=generator, dtype=torch.float64)
    aperture = 0.2

    centered = activations - activations.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / centered.shape[0]
    identity = torch.eye(centered.shape[1], dtype=torch.float64)
    dense = covariance @ torch.linalg.inv(covariance + aperture**-2 * identity)
    expected = torch.linalg.eigvalsh(dense).sort(descending=True).values.numpy()

    actual = conceptor_eigenvalues(activations, aperture)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)


def test_conceptor_eigenvalues_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        conceptor_eigenvalues(torch.ones(3), 0.1)
    with pytest.raises(ValueError, match="finite and positive"):
        conceptor_eigenvalues(torch.ones(3, 2), 0.0)
    values = torch.ones(3, 2)
    values[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        conceptor_eigenvalues(values, 0.1)


def test_effective_rank_definitions_are_distinct_and_named() -> None:
    spectrum = [0.9, 0.1, 0.0]
    shannon = shannon_effective_rank(spectrum)
    participation = participation_ratio_effective_rank(spectrum)
    expected_shannon = np.exp(-(0.9 * np.log(0.9) + 0.1 * np.log(0.1)))
    assert shannon == pytest.approx(expected_shannon)
    assert participation == pytest.approx(1 / (0.9**2 + 0.1**2))
    assert shannon > participation
    assert shannon_effective_rank([1, 1, 1]) == pytest.approx(3.0)
    assert participation_ratio_effective_rank([1, 1, 1]) == pytest.approx(3.0)


def test_user_checkpoint_requires_a_matching_run_manifest() -> None:
    specification = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "experimental_materials_sha256": "materials",
    }
    identity = hashlib.sha256(
        json.dumps(
            specification, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "run_specification": specification,
        "run_specification_sha256": identity,
    }
    assert (
        validate_run_binding(
            manifest,
            identity,
            required_model="Qwen/Qwen2.5-1.5B-Instruct",
            current_materials_sha256="materials",
        )
        == identity
    )
    with pytest.raises(RuntimeError, match="not owned"):
        validate_run_binding(
            manifest,
            "different",
            required_model="Qwen/Qwen2.5-1.5B-Instruct",
            current_materials_sha256="materials",
        )
    with pytest.raises(RuntimeError, match="materials"):
        validate_run_binding(
            manifest,
            identity,
            required_model="Qwen/Qwen2.5-1.5B-Instruct",
            current_materials_sha256="different",
        )


def test_post_inspection_direction_does_not_halve_p_value() -> None:
    semantic = [7, 8, 9, 10, 11, 12]
    persona = [0, 1, 2, 3, 4, 5, 6]
    statistic, p_value, direction = oriented_two_sided_mann_whitney(
        semantic, persona
    )

    assert statistic == 42
    assert direction == "sem>per"
    assert p_value == pytest.approx(0.0011655011655011655)
    assert p_value > 0.05 / 48


def test_complete_ordering_is_direction_agnostic_and_rejects_ties() -> None:
    assert is_complete_ordering([3, 4], [1, 2])
    assert is_complete_ordering([1, 2], [3, 4])
    assert not is_complete_ordering([1, 2], [2, 3])


def test_formula_failures_are_not_silently_converted_to_nan() -> None:
    with pytest.raises(RuntimeError, match="broken"):
        evaluate_finite_formula(
            "broken", "model", "concept", lambda *_: 1 / 0, None
        )
    with pytest.raises(ValueError, match="non_finite"):
        evaluate_finite_formula(
            "non_finite", "model", "concept", lambda *_: float("nan"), None
        )


@pytest.mark.parametrize("first,second,status", [
    ([0] * 6, [0] * 7, "equal_constant_groups"),
    ([1] * 6, [2] * 7, "zero_variance_negative_difference"),
    ([2] * 6, [1] * 7, "zero_variance_positive_difference"),
    ([0.1] * 6, [0.1] * 7, "equal_constant_groups"),
    ([0.1] * 6, [0.2] * 7, "zero_variance_negative_difference"),
    ([0.2] * 6, [0.1] * 7, "zero_variance_positive_difference"),
])
def test_degenerate_effects_are_explicit_not_finite_sentinels(first, second, status):
    value, actual_status = hedges_g(first, second)
    assert value is None
    assert actual_status == status
    assert json.loads(json.dumps({"g": value}, allow_nan=False))["g"] is None


def test_hedges_g_matches_definition_and_preserves_direction():
    first, second = np.arange(6.0), np.arange(7.0) + 3
    df = len(first) + len(second) - 2
    pooled = np.sqrt(((len(first) - 1) * first.var(ddof=1) +
                      (len(second) - 1) * second.var(ddof=1)) / df)
    expected = (1 - 3 / (4 * df - 1)) * (first.mean() - second.mean()) / pooled
    assert hedges_g(first, second)[0] == pytest.approx(expected)
    assert hedges_g(first, second)[1] == "defined"
    assert hedges_g(second, first)[0] == pytest.approx(-expected)


def test_mann_whitney_ties_use_asymptotic_not_no_tie_exact_bound():
    assert mann_whitney_auto_method(range(6), range(6, 13)) == "exact"
    first, second = [1] * 6, [0] * 7
    assert mann_whitney_auto_method(first, second) == "asymptotic"
    _, p_value, _ = oriented_two_sided_mann_whitney(first, second)
    assert p_value == pytest.approx(0.0007205896319716217)
    assert p_value < 0.05 / 48


def test_spectrum_summaries_share_the_actual_spectrum():
    eigenvalues = np.array([0.4, 0.2, 0.1, 0])
    summary = spectrum_summary(eigenvalues)
    assert summary["rawCCR"] == pytest.approx(eigenvalues.mean())
    assert summary["SPR"] == pytest.approx(np.square(eigenvalues).mean())
    assert summary["SCR"] == pytest.approx(np.square(eigenvalues).sum() / eigenvalues.sum())
    assert summary["EffRank"] == pytest.approx(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


def test_variance_edge_uses_actual_sample_count_and_width():
    eigenvalues = np.array([0.4, 0.2, 0.1, 0])
    values = []
    for n in (4, 100):
        edge = eigenvalues[:3].var() * (1 + np.sqrt(n / len(eigenvalues))) ** 2
        actual = spectral_variance_edge_fraction(eigenvalues, n)
        assert actual == pytest.approx(np.mean(eigenvalues > edge))
        values.append(actual)
    assert values[0] != values[1]
    for invalid in (0, -1, True, 4.5):
        with pytest.raises(ValueError, match="positive integer"):
            spectral_variance_edge_fraction(eigenvalues, invalid)


SCREEN = Path(__file__).with_name("formula_screening.py")
ROOT = SCREEN.parents[3]


def run_screen(*arguments):
    result = subprocess.run([sys.executable, str(SCREEN), *map(str, arguments)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout


def test_historical_summary_mode_needs_no_checkpoint(tmp_path):
    output = tmp_path / "historical.json"
    run_screen("--mode", "historical-summaries", "--output", output)
    document = json.loads(output.read_text())
    assert len(document["formulas"]) == 13
    assert document["config"]["bonferroni_family_size"] == 48
    assert set(document["config"]["input_sha256"]) == {"consolidated_json"}
    assert document["config"]["steering_associations_available"] is True
    assert document["config"]["run_specification_sha256"] is None
    original = json.loads((ROOT / "provenance/broad_study_results.json").read_text())
    scr = next(row for row in document["formulas"] if row["name"] == "SCR")
    assert scr["values_1.5B"] == {
        name: values["SCR"]
        for name, values in original["Qwen2.5-1.5B"]["results"]["theory"].items()
    }
    frozen = json.loads((SCREEN.parents[1] / "results/formula_screening_results.json").read_text())
    frozen_rows = {row["name"]: row for row in frozen["formulas"]}
    for row in document["formulas"]:
        for name, value in row["values_1.5B"].items():
            assert value == pytest.approx(frozen_rows[row["name"]]["values_1.5B"][name],
                                          rel=1e-12, abs=1e-12)


def test_checkpoint_mode_uses_only_checkpoint_spectra(tmp_path):
    # Target names are the screen's fixed scientific inventory, independent of JSON results.
    semantic = ['antonym', 'country_capital', 'present_past', 'english_french',
                'singular_plural', 'capitalize']
    persona = ['sycophancy', 'hallucination', 'evil', 'impolite', 'optimistic',
               'apathetic', 'humorous']
    specification = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "experimental_materials_sha256": hashlib.sha256(
            (ROOT / "experimental_materials.json").read_bytes()).hexdigest(),
        "fast_mode": True,
        "num_icl_prompts": 50,
    }
    identity = hashlib.sha256(json.dumps(specification, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"run_specification": specification,
                                    "run_specification_sha256": identity}))
    generator = torch.Generator().manual_seed(4)
    activations = {name: torch.randn(8 + i, 12, generator=generator) * (i + 1) / 10
                   for i, name in enumerate(semantic + persona)}
    checkpoint = tmp_path / "extractions.pt"
    torch.save({"run_specification_sha256": identity,
                "semantic": {name: {0: activations[name]} for name in semantic},
                "persona": {name: {"pos": {0: activations[name]}} for name in persona}},
               checkpoint)
    output = tmp_path / "checkpoint.json"
    args = ["--extractions", checkpoint, "--extractions-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--run-manifest", manifest, "--output", output]
    run_screen(*args)
    document = json.loads(output.read_text())
    assert len(document["formulas"]) == 48
    assert set(document["config"]["input_sha256"]) == {"extractions", "run_manifest"}
    assert document["config"]["steering_associations_available"] is False
    assert document["summary"]["steering_nominal_p_lt_05_descriptive"] is None
    assert document["summary"]["u42_on_all_models"] is None
    assert document["config"]["fit_sample_counts"] == {
        name: values.shape[0] for name, values in activations.items()}
    assert set(document["config"]["fit_dimensions"].values()) == {12}
    rows = {row["name"]: row for row in document["formulas"]}
    assert all(row["rho_steering"] is None and row["p_steering"] is None
               and row["cross_model_rho"] is None and row["all_models_u42"] is None
               for row in rows.values())
    for name, values in activations.items():
        spectrum = conceptor_eigenvalues(values, 0.05)
        assert rows["SCR"]["values_1.5B"][name] == pytest.approx(spectrum_summary(spectrum)["SCR"])
        assert rows["MP_Fraction"]["values_1.5B"][name] == pytest.approx(
            spectral_variance_edge_fraction(spectrum, values.shape[0]))
    assert rows["N_above_0.5"]["hedges_g"] is None
    assert rows["N_above_0.5"]["hedges_g_status"] == "equal_constant_groups"
    result = subprocess.run([sys.executable, str(SCREEN), *map(str, args[:-1]),
                             str(tmp_path / "mixed.json"), "--consolidated-json",
                             str(ROOT / "provenance/broad_study_results.json")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "cannot combine" in result.stderr
    assert not (tmp_path / "mixed.json").exists()
