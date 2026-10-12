from __future__ import annotations

import copy
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "provenance" / "historical_environment.json"
RESULTS = ROOT / "experiments" / "matched_control" / "results"

TOP_LEVEL_KEYS = {
    "schema_version",
    "applies_to",
    "models",
    "generation_software",
    "hardware",
    "decoding",
    "hooks",
    "historical_judge",
    "note",
}
MODEL_KEYS = {"model_id", "revision", "layer", "num_layers", "hidden_size"}
EXPECTED_MODELS = {
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", 9),
    "7b": ("Qwen/Qwen2.5-7B-Instruct", 9),
    "8b": ("meta-llama/Llama-3.1-8B-Instruct", 10),
}
EXPECTED_DECODING = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.95,
    "do_sample": True,
}
DECODING_KEYS = set(EXPECTED_DECODING) | {"base_seed", "seed_rule", "condition_order_seed"}
CORRECTED_COHERENCE_MODEL = "deepseek-v4-pro"
HEX40 = re.compile(r"[0-9a-f]{40}")
PATH_LIKE = re.compile(
    r"^[/~]|\\|[A-Za-z]:[\\/]|(?:^|/)(?:Users|home|content|mnt|tmp|var)/|"
    r"(?:^|[\s(])/[A-Za-z]|@"
)
BANNED_KEY_FRAGMENTS = ("path", "host", "user", "unix", "timestamp", "created")


def _strings(value, keys: bool = True):
    if isinstance(value, dict):
        for key, item in value.items():
            if keys:
                yield key
            yield from _strings(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item, keys)
    elif isinstance(value, str):
        yield value


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def validate(record: dict) -> list[str]:
    errors = []
    if set(record) != TOP_LEVEL_KEYS:
        errors.append(f"top-level keys {sorted(set(record) ^ TOP_LEVEL_KEYS)}")
    models = record.get("models", {})
    if set(models) != set(EXPECTED_MODELS):
        errors.append("model keys")
    for model_key, (model_id, layer) in EXPECTED_MODELS.items():
        entry = models.get(model_key, {})
        if set(entry) != MODEL_KEYS:
            errors.append(f"{model_key} keys")
        if entry.get("model_id") != model_id or entry.get("layer") != layer:
            errors.append(f"{model_key} id or layer")
        if not HEX40.fullmatch(str(entry.get("revision", ""))):
            errors.append(f"{model_key} revision")
    decoding = record.get("decoding", {})
    if set(decoding) != DECODING_KEYS:
        errors.append("decoding keys")
    for key, expected in EXPECTED_DECODING.items():
        if decoding.get(key) != expected or type(decoding.get(key)) is not type(expected):
            errors.append(f"decoding {key}")
    judge = record.get("historical_judge", {})
    if judge.get("model") != "gemini-2.5-flash" or "deepseek" in json.dumps(judge).lower():
        errors.append("historical judge must be the Gemini judge")
    allowed_slash = {model_id for model_id, _ in EXPECTED_MODELS.values()}
    for text in _strings(record):
        if text in allowed_slash:
            continue
        if "/" in text or PATH_LIKE.search(text):
            errors.append(f"path-like string {text!r}")
    for key in _keys(record):
        if any(fragment in key.lower() for fragment in BANNED_KEY_FRAGMENTS):
            errors.append(f"banned key {key!r}")
    return errors


class HistoricalEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw = RECORD.read_text(encoding="utf-8")
        cls.raw = raw
        cls.record = json.loads(raw)

    def test_record_validates(self) -> None:
        self.assertEqual(validate(self.record), [])

    def test_two_space_formatting(self) -> None:
        self.assertTrue(self.raw.endswith("\n"))
        self.assertTrue(self.raw.startswith('{\n  "schema_version"'))
        for line in self.raw.splitlines():
            indent = len(line) - len(line.lstrip(" "))
            self.assertEqual(indent % 2, 0, line)
            self.assertNotIn("\t", line)

    def test_scope_excludes_other_studies(self) -> None:
        scope = self.record["applies_to"]
        self.assertEqual(scope["study"], "matched 1,024-token panel")
        self.assertEqual(scope["n_per_model_target_condition"], 40)
        self.assertEqual(
            scope["responses_per_model"],
            len(scope["targets"]) * len(scope["conditions"]) * scope["n_per_model_target_condition"],
        )
        excluded = " ".join(scope["not_applicable_to"]).lower()
        for study in ("broad seven-target", "protocol swap", "extraction", "finite alternatives"):
            self.assertIn(study, excluded)

    def test_hooks_and_software(self) -> None:
        self.assertEqual(self.record["hooks"]["layer_indexing"], "zero-based")
        self.assertIs(self.record["hooks"]["all_sequence_positions"], True)
        software = self.record["generation_software"]
        self.assertEqual(software["dtype"], "float16")
        self.assertIs(software["deterministic_algorithms_enabled"], False)
        self.assertTrue(software["torch"].endswith("+cu" + software["torch_cuda"].replace(".", "")))

    def test_judge_is_not_the_corrected_coherence_endpoint(self) -> None:
        judge = self.record["historical_judge"]
        self.assertNotEqual(judge["model"], CORRECTED_COHERENCE_MODEL)
        self.assertEqual(judge["temperature"], 0.0)
        self.assertEqual(judge["thinking_budget"], 0)
        self.assertEqual(set(judge["views"]), {"prefix200", "full"})
        corrected = ROOT / "experiments" / "corrected_coherence" / "README.md"
        if corrected.is_file():
            text = corrected.read_text(encoding="utf-8").lower()
            self.assertIn(CORRECTED_COHERENCE_MODEL, text)

    def test_consistent_with_shipped_matched_control_results(self) -> None:
        targets = set(self.record["applies_to"]["targets"])
        aperture = self.record["hooks"]["aperture"]
        for model_key, entry in self.record["models"].items():
            quality = json.loads(
                (RESULTS / model_key / "analysis_data_quality.json").read_text(encoding="utf-8")
            )
            geometry = json.loads((RESULTS / model_key / "geometry.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["scipy_version"], self.record["generation_software"]["scipy"])
            self.assertEqual(quality["expected_response_rows"], self.record["applies_to"]["responses_per_model"])
            self.assertEqual(set(geometry), targets, model_key)
            for target, values in geometry.items():
                self.assertEqual(values["layer"], entry["layer"], (model_key, target))
                self.assertEqual(values["hidden_size"], entry["hidden_size"], (model_key, target))
                self.assertEqual(values["aperture"], aperture, (model_key, target))

    def test_validator_rejects_extra_top_level_key(self) -> None:
        mutated = copy.deepcopy(self.record)
        mutated["raw_manifest"] = {}
        self.assertTrue(any("top-level" in error for error in validate(mutated)))
        mutated = copy.deepcopy(self.record)
        mutated["decoding"]["historical_token_ids_available"] = False
        self.assertIn("decoding keys", validate(mutated))

    def test_validator_rejects_malformed_revision(self) -> None:
        for revision in ("main", "989aa7980e4cf806f80c7fef2b1adb7bc71aa30", "989AA7980E4CF806F80C7FEF2B1ADB7BC71AA306"):
            mutated = copy.deepcopy(self.record)
            mutated["models"]["1.5b"]["revision"] = revision
            self.assertIn("1.5b revision", validate(mutated), revision)

    def test_validator_rejects_paths_and_wrong_judge(self) -> None:
        mutated = copy.deepcopy(self.record)
        mutated["note"] = "copied from " + "/" + "content/drive/run"
        self.assertTrue(any("path-like" in error for error in validate(mutated)))
        mutated = copy.deepcopy(self.record)
        mutated["historical_judge"]["model"] = CORRECTED_COHERENCE_MODEL
        self.assertIn("historical judge must be the Gemini judge", validate(mutated))
        mutated = copy.deepcopy(self.record)
        mutated["hardware"]["created_unix"] = 1.0
        self.assertTrue(any("banned key" in error for error in validate(mutated)))


if __name__ == "__main__":
    unittest.main()
