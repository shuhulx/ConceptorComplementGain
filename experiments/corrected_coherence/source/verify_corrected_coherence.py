#!/usr/bin/env python3
"""Offline verification of the corrected matched-panel coherence evidence.

Uses only files shipped in this repository; never imports provider code or
contacts a network service. Exits non-zero if any check fails.

Evidence levels printed with each check:
  RECOMPUTED  recomputed here from the shipped question-level table
  BOUND       cross-file consistency
  INTEGRITY   recorded aggregate diagnostics; individual judgments are not distributed
"""

from __future__ import annotations

import json
import math

import corrected_coherence as cc

TOLERANCE = 1e-12


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=TOLERANCE)


def checks():
    audit = json.loads(cc.CORRECTED_AUDIT.read_text(encoding="utf-8"))

    corrected = cc.load_corrected_results()
    table = cc.load_question_contrasts()
    yield "RECOMPUTED", "question table: 12 settings x 20 questions x 2 pairs = 480 pairs", (
        sum(len(v) for v in table.values()) * cc.PAIRS_PER_QUESTION == 480)
    recomputed = {(r["model"], r["trait"]): r for r in cc.summarize_questions(table)}
    for field in ("n_pairs", "n_questions", "bootstrap_seed", "mean_difference", "ci95_low",
                  "ci95_high", "wilcoxon_p_nominal", "holm_12_corrected_coherence"):
        yield "RECOMPUTED", f"12 corrected contrasts: {field}", all(
            close(recomputed[key][field], corrected[key][field]) for key in cc.SETTINGS)
    yield "RECOMPUTED", "12-test Holm family across all models; all adjusted p = 1", (
        len(recomputed) == cc.HOLM_FAMILY_SIZE
        and all(r["holm_12_corrected_coherence"] == 1.0 for r in recomputed.values()))
    yield "RECOMPUTED", "all 12 corrected 95% intervals include zero", all(
        r["ci95_low"] <= 0 <= r["ci95_high"] for r in recomputed.values())
    means = [r["mean_difference"] for r in recomputed.values()]
    yield "RECOMPUTED", "mean contrast range -2.125 .. +1.875", (
        (min(means), max(means)) == (-2.125, 1.875) == tuple(audit["mean_contrast_range"]))
    yield "RECOMPUTED", "seven zero-width intervals (all question differences zero)", sum(
        not any(table[key]) and recomputed[key]["ci95_low"] == recomputed[key]["ci95_high"] == 0
        for key in cc.SETTINGS) == 7

    historical = cc.historical_gemini_coherence()
    yield "BOUND", "historical_mean_difference equals Gemini endpoint=coherence rows", all(
        corrected[key]["historical_mean_difference"] == historical[key]["mean_difference"]
        for key in cc.SETTINGS)
    yield "RECOMPUTED", "change_in_contrast = corrected - historical", all(
        close(corrected[key]["change_in_contrast"],
              recomputed[key]["mean_difference"] - historical[key]["mean_difference"])
        for key in cc.SETTINGS)
    endpoints = None
    try:
        endpoints = cc.current_matched_endpoints()
        cc.validate_current_endpoints(endpoints)
        separated = all(row["source"] == cc.CURRENT_TARGET_SOURCE
                        for row in endpoints["target_expression"].values())
    except ValueError:
        endpoints = None
        separated = False
    yield "BOUND", "current endpoints: 12 Gemini trait rows + 12 corrected coherence", separated
    yield "BOUND", "historical Gemini coherence is labelled historical only", all(
        row["status"] == "historical_gemini_old_rubric" for row in historical.values())

    secondary = cc.secondary_evaluator_sensitivity()
    pairs = sum(int(row["selected_pair_n"]) for row in secondary.values())
    yield "BOUND", "secondary subset kept separate: 60 pairs / 240 answers, not current", (
        pairs == 60 and pairs * 4 == 240
        and all(row["status"] == "historical_secondary_subset" for row in secondary.values())
        and endpoints is not None
        and sum(row["n_pairs"] for row in endpoints["coherence"].values()) == 480)

    settings = {(r["model"], r["trait"]): r for r in audit["settings"]}
    yield "BOUND", "recorded per-setting contrasts equal corrected results", set(settings) == set(
        cc.SETTINGS) and all(close(settings[k][f], corrected[k][f]) for k in cc.SETTINGS for f in (
            "mean_difference", "ci95_low", "ci95_high", "wilcoxon_p_nominal",
            "holm_12_corrected_coherence"))
    yield "INTEGRITY", "recorded: 960 responses, 480 pairs, 960 first attempts, 0 failures", (
        (audit["responses"], audit["pairs"], audit["attempts"], audit["failed_attempts"])
        == (960, 480, 960, 0))
    yield "INTEGRITY", "recorded: deepseek-v4-pro, temperature 0, thinking disabled, 32 tokens", (
        audit["requested_model"] == "deepseek-v4-pro"
        and audit["returned_models"] == ["deepseek-v4-pro"] and audit["temperature"] == 0
        and audit["thinking"] == "disabled" and audit["max_tokens"] == 32)
    histogram = audit["score_histogram"]
    yield "INTEGRITY", "recorded: 872/960 answers scored 100 (histogram sums to 960)", (
        sum(histogram.values()) == 960 and histogram["100.0"] == audit["ceiling_responses"] == 872
        and sum(r["corrected_ceiling_n_of_80"] for r in audit["settings"]) == 872)
    all_ceiling = {k for k, r in settings.items() if r["corrected_ceiling_n_of_80"] == 80}
    yield "INTEGRITY", "recorded: seven all-100 settings, each with a zero question table", (
        len(all_ceiling) == 7 and all(not any(table[key]) for key in all_ceiling))
    yield "INTEGRITY", "recorded: 195 identical-text pairs, 2 disagreements (10 and 20 points)", (
        audit["identical_question_answer_pairs"] == 195
        and audit["identical_pair_score_disagreements"] == 2
        and audit["identical_pair_max_absolute_score_difference"] == 20.0
        and close(audit["identical_pair_mean_absolute_score_difference"] * 195, 30.0))
    cap = audit["cap_sensitivity"]
    yield "INTEGRITY", "recorded: 2 cap-affected 1.5b hallucination pairs; mean 1.875 -> 1.375", (
        len(cap) == 1 and (cap[0]["model"], cap[0]["trait"]) == ("1.5b", "hallucination")
        and (cap[0]["excluded_pairs"], cap[0]["remaining_pairs"]) == (2, 38)
        and cap[0]["equal_question_mean"] == 1.375
        and corrected["1.5b", "hallucination"]["mean_difference"] == 1.875)
    yield "INTEGRITY", "recorded: arm means differ by the corrected contrast", all(
        close(settings[k]["complete_mean_score"] - settings[k]["control_mean_score"],
              corrected[k]["mean_difference"]) for k in cc.SETTINGS)


def main() -> int:
    print("Evidence levels: RECOMPUTED from shipped question table; BOUND = cross-file "
          "consistency; INTEGRITY = recorded summaries, not a check of individual judgments.")
    failures = 0
    for level, name, passed in checks():
        failures += not passed
        print(f"{'PASS' if passed else 'FAIL'} [{level}] {name}")
    print(f"\n{'FAILED' if failures else 'PASSED'}: corrected coherence verification "
          f"({failures} failure{'s' if failures != 1 else ''})")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError) as error:
        print(f"FAIL [ERROR] {type(error).__name__}: {error}")
        raise SystemExit(1)
