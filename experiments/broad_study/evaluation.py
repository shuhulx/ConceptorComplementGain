"""Input validation and hosted-evaluator helpers for the broad study."""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.broad_study.checkpointing import (
    atomic_write_json,
    canonical_json_hash,
    require_checkpoint_identity,
)

PersonaKey = tuple[str, str, int]
PERSONA_ROW_FIELDS = frozenset({"question", "response", "trait_score", "coherence"})
PERSONA_TRAIT_NAMES = frozenset({
    "sycophancy", "hallucination", "evil", "impolite", "optimistic", "apathetic", "humorous",
})

# Documented response grammar of the broad study rubrics in experimental_materials.json:
# "Respond with a number between 0 and 100. Don't say anything else, just the
# number." The whole stripped reply must be one unsigned decimal number; signs,
# exponents, units, fractions such as "80/100", and surrounding prose are
# outside the grammar. Decimals are kept exactly (15.5 stays 15.5).
JUDGE_SCORE_GRAMMAR = re.compile(r"[0-9]+(?:\.[0-9]+)?", re.ASCII)

# The broad study records the trait-rubric token REFUSAL as score 0. Matched-control
# analysis instead treats refusal as a separate categorical outcome without a
# numeric trait score. The coherence rubric does not document REFUSAL.
LEGACY_BROAD_STUDY_REFUSAL_TRAIT_SCORE = 0.0


class InvalidJudgeScore(ValueError):
    """Hosted-evaluator text outside the documented response grammar."""


def _is_string_list(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, str) for item in value)
    )


def require_function_vector_document(document: Any) -> Mapping[str, Any]:
    """Return the "functions" mapping of function_vectors.json, 32 pairs per task.

    Pair contents and the task set are checked separately by
    fetch_function_vectors.require_expected_pairs.
    """
    functions = document.get("functions") if isinstance(document, Mapping) else None
    if not isinstance(functions, Mapping):
        raise ValueError(
            "function_vectors.json must be a JSON object with a 'functions' mapping"
        )
    for task, specification in functions.items():
        pairs = specification.get("pairs") if isinstance(specification, Mapping) else None
        if not isinstance(pairs, list) or len(pairs) != 32:
            raise ValueError(f"Function Vector task {task} must contain 32 pairs")
    return functions


def require_experimental_materials(document: Any) -> None:
    """Raise unless experimental_materials.json has every field run_study.py reads."""
    traits = document.get("traits") if isinstance(document, Mapping) else None
    if not isinstance(traits, Mapping) or not traits:
        raise ValueError(
            "experimental_materials.json must be a JSON object with a 'traits' mapping"
        )
    if not isinstance(document.get("coherence_prompt"), str):
        raise ValueError("experimental_materials.json has no 'coherence_prompt' string")
    # The figures assume these seven targets; the judge self-test uses sycophancy.
    if set(traits) != PERSONA_TRAIT_NAMES:
        raise ValueError(
            "experimental_materials.json must define exactly the seven persona traits "
            f"{sorted(PERSONA_TRAIT_NAMES)}"
        )
    for name, trait in traits.items():
        if not isinstance(trait, Mapping):
            raise ValueError(f"Trait {name} must be a JSON object")
        instructions = trait.get("instructions")
        if not isinstance(instructions, list) or len(instructions) != 5 or not all(
            isinstance(pair, Mapping)
            and isinstance(pair.get("pos"), str)
            and isinstance(pair.get("neg"), str)
            for pair in instructions
        ):
            raise ValueError(
                f"Trait {name} must contain 5 instruction pairs with 'pos' and 'neg' strings"
            )
        if not _is_string_list(trait.get("extract_questions"), 20):
            raise ValueError(f"Trait {name} must contain 20 extraction questions")
        if not _is_string_list(trait.get("eval_questions"), 20):
            raise ValueError(f"Trait {name} must contain 20 evaluation questions")
        if not isinstance(trait.get("eval_prompt"), str):
            raise ValueError(f"Trait {name} is missing its evaluation rubric")


def prompt_block_reason(response: Any) -> str | None:
    feedback = getattr(response, "prompt_feedback", None)
    raw_reason = getattr(feedback, "block_reason", None)
    if raw_reason is None:
        return None
    reason = str(raw_reason)
    if not reason or reason.lower() == "none" or "UNSPECIFIED" in reason.upper():
        return None
    return reason


def parse_judge_score(text: Any, *, legacy_refusal_as_zero: bool = False) -> float:
    """Parse one hosted-evaluator reply; raise InvalidJudgeScore otherwise."""
    if not isinstance(text, str):
        raise InvalidJudgeScore(f"Judge reply is not text: {type(text).__name__}")
    stripped = text.strip()
    if legacy_refusal_as_zero and stripped.upper() == "REFUSAL":
        return LEGACY_BROAD_STUDY_REFUSAL_TRAIT_SCORE
    if JUDGE_SCORE_GRAMMAR.fullmatch(stripped) is None:
        raise InvalidJudgeScore(f"Judge reply is outside the score grammar: {text!r}")
    value = float(stripped)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise InvalidJudgeScore(f"Judge score is outside [0, 100]: {text!r}")
    return value


def require_complete_evaluation(stats: Mapping[str, int]) -> None:
    safety_blocks = int(stats.get("safety_blocks", 0))
    errors = int(stats.get("errors", 0))
    empty = int(stats.get("empty", 0))
    invalid = int(stats.get("invalid", 0))
    if safety_blocks + errors + empty + invalid:
        raise RuntimeError(
            "Automated evaluation was incomplete: "
            f"safety_blocks={safety_blocks}, errors={errors}, empty={empty}, "
            f"invalid={invalid}. "
            "No partial evaluator aggregate will be reported."
        )


def judgments_are_complete(rows: Mapping[object, Mapping[str, Any]]) -> bool:
    """Field completeness only: every present row carries both scores."""
    return bool(rows) and all(
        row.get("trait_score") is not None and row.get("coherence") is not None
        for row in rows.values()
    )


def persona_response_universe(
    trait_names: Iterable[str], condition_names: Iterable[str], n_samples: int
) -> tuple[PersonaKey, ...]:
    """Exact (trait, condition, sample_index) keys of the active configuration."""
    traits = list(trait_names)
    conditions = list(condition_names)
    if len(set(traits)) != len(traits) or len(set(conditions)) != len(conditions):
        raise ValueError("Duplicate trait or condition name in the active configuration")
    if type(n_samples) is not int or n_samples <= 0:
        raise ValueError(f"Invalid sample count: {n_samples!r}")
    return tuple(
        (trait, condition, index)
        for trait in traits
        for condition in conditions
        for index in range(n_samples)
    )


def reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """json.load object_pairs_hook that refuses silently overwritten keys."""
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        document[key] = value
    return document


def parse_persona_response_key(text: Any) -> PersonaKey:
    """Parse the canonical str((trait, condition, sample_index)) checkpoint key."""
    try:
        value = ast.literal_eval(text) if isinstance(text, str) else None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        value = None
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or not isinstance(value[0], str)
        or not isinstance(value[1], str)
        or type(value[2]) is not int
        or str(value) != text
    ):
        raise RuntimeError(f"Malformed persona response key: {text!r}")
    return value


def decode_persona_checkpoint_responses(responses: Any) -> dict[PersonaKey, Any]:
    if not isinstance(responses, Mapping):
        raise RuntimeError("Persona checkpoint has no response mapping")
    decoded: dict[PersonaKey, Any] = {}
    for text, row in responses.items():
        key = parse_persona_response_key(text)
        if key in decoded:
            raise RuntimeError(f"Duplicate persona response key: {text!r}")
        decoded[key] = row
    return decoded


def missing_persona_responses(
    rows: Mapping[Any, Any],
    expected_keys: Iterable[PersonaKey],
    eval_questions: Mapping[str, Sequence[str]],
) -> set[PersonaKey]:
    """Validate present rows against the universe; return the missing keys.

    Raises on extra or malformed keys, malformed rows, and a question that is
    not the configured eval_questions[sample_index % len(eval_questions)].
    """
    expected = set(expected_keys)
    extra = set(rows) - expected
    if extra:
        raise RuntimeError(
            f"Persona responses contain {len(extra)} keys outside the active "
            f"configuration, e.g. {sorted(map(repr, extra))[:3]}"
        )
    for key, row in rows.items():
        trait, _condition, sample_index = key
        if (
            not isinstance(row, Mapping)
            or set(row) != PERSONA_ROW_FIELDS
            or not isinstance(row["response"], str)
        ):
            raise RuntimeError(f"Malformed persona response row: {key!r}")
        questions = eval_questions[trait]
        if row["question"] != questions[sample_index % len(questions)]:
            raise RuntimeError(f"Persona response has the wrong question binding: {key!r}")
    return expected - set(rows)


def checkpointed_persona_traits(
    rows: Mapping[Any, Any],
    expected_keys: Iterable[PersonaKey],
    eval_questions: Mapping[str, Sequence[str]],
) -> set[str]:
    """Return completed traits for resume; reject incomplete traits."""
    missing = missing_persona_responses(rows, expected_keys, eval_questions)
    present = {key[0] for key in rows}
    partial = present & {key[0] for key in missing}
    if partial:
        raise RuntimeError(
            f"Persona checkpoint has incomplete traits {sorted(partial)}; "
            "refusing to resume from a partial trait"
        )
    return present


def require_complete_persona_corpus(
    rows: Mapping[Any, Any],
    expected_keys: Iterable[PersonaKey],
    eval_questions: Mapping[str, Sequence[str]],
) -> None:
    expected = tuple(expected_keys)
    missing = missing_persona_responses(rows, expected, eval_questions)
    if missing:
        raise RuntimeError(
            f"Persona response corpus is incomplete: {len(missing)} of "
            f"{len(expected)} configured rows are missing, "
            f"e.g. {sorted(missing)[:3]}"
        )


def _is_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 100.0
    )


def study_judgments_are_complete(
    rows: Mapping[Any, Mapping[str, Any]], expected_keys: Iterable[PersonaKey]
) -> bool:
    """Study completeness: the exact configured universe, each row fully scored."""
    return (
        set(rows) == set(expected_keys)
        and judgments_are_complete(rows)
        and all(
            _is_score(row["trait_score"]) and _is_score(row["coherence"])
            for row in rows.values()
        )
    )


def score_persona_responses(
    rows: Mapping[PersonaKey, dict[str, Any]],
    *,
    trait_prompts: Mapping[str, str],
    coherence_prompt: str,
    judge: Any,
    evaluator_specification: Mapping[str, Any],
    checkpoint_path: Path,
    run_specification_sha256: str,
) -> int:
    """Resume judgments for this exact corpus, saving each successful score.

    Existing scores in ``rows`` are not trusted without the bound checkpoint.
    Invalid or missing checkpoint scores are retried. The caller must still
    require complete judgments before reporting an aggregate.
    """
    inputs = {
        "responses": {
            str(key): {"question": row["question"], "response": row["response"]}
            for key, row in rows.items()
        },
        "trait_prompts": dict(trait_prompts),
        "coherence_prompt": coherence_prompt,
        "evaluator": dict(evaluator_specification),
        "trait_refusal_score": LEGACY_BROAD_STUDY_REFUSAL_TRAIT_SCORE,
    }
    input_hash = canonical_json_hash(inputs)
    saved_scores: dict[str, Any] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as handle:
            saved = json.load(handle, object_pairs_hook=reject_duplicate_json_keys)
        if not isinstance(saved, dict):
            raise RuntimeError("Malformed judgment checkpoint")
        require_checkpoint_identity(saved, run_specification_sha256, checkpoint_path)
        if saved.get("evaluation_inputs_sha256") != input_hash:
            raise RuntimeError("Judgment checkpoint inputs or evaluator do not match")
        saved_scores = saved.get("scores")
        if not isinstance(saved_scores, dict) or set(saved_scores) - set(inputs["responses"]):
            raise RuntimeError("Judgment checkpoint has invalid response keys")
        if any(
            not isinstance(scores, dict)
            or set(scores) - {"trait_score", "coherence"}
            for scores in saved_scores.values()
        ):
            raise RuntimeError("Malformed judgment checkpoint scores")

    # Normalize missing/invalid scores before either reuse or persistence.
    for key, row in rows.items():
        prior = saved_scores.get(str(key), {})
        for field in ("trait_score", "coherence"):
            row[field] = prior.get(field) if _is_score(prior.get(field)) else None
    scores = {
        str(key): {field: row[field] for field in ("trait_score", "coherence")}
        for key, row in rows.items()
    }
    calls = 0
    for key, row in rows.items():
        for field, prompt, refusal in (
            ("trait_score", trait_prompts[key[0]], True),
            ("coherence", coherence_prompt, False),
        ):
            if row[field] is not None:
                continue
            value = judge.judge_score(
                prompt, row["question"], row["response"],
                legacy_refusal_as_zero=refusal,
            )
            calls += 1
            if not _is_score(value):
                continue
            row[field] = value
            scores[str(key)][field] = value
            atomic_write_json(checkpoint_path, {
                "run_specification_sha256": run_specification_sha256,
                "evaluation_inputs_sha256": input_hash,
                "scores": scores,
            })
    return calls
