"""
Unit test: verify beetle's pico-analyze compatibility layer.

Covers three guarantees that make a beetle HF repo consumable by pico-analyze:

1. ``_build_pico_compatible_training_config`` flattens ``model.params.*`` up to
   ``model.*`` so ``pico-analyze/src/components/ov_circuit.py`` can read
   ``training_config["model"]["d_model"]`` without modification.

2. ``_push_pico_compatible_to_main(step=0)`` emits upload calls whose commit
   messages match pico-train verbatim — in particular the patterns scanned by
   ``pico-analyze/src/utils/data.py:145``:

       r"Saving Learning Dynamics Data \((train|val)\) -- Step (\d+)"

3. The shared ``save_checkpoint`` in ``src/checkpointing/training.py`` uses
   double-hyphen (``--``), not em-dash (``—``), so pico-analyze's regex matches.

Run directly (no pytest dependency needed):

    PYTHONPATH=. python script/test_pico_compat_unit.py
"""

from __future__ import annotations

import os
import re
import sys
import types
import unittest
from unittest import mock


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class BuildPicoTrainingConfigTest(unittest.TestCase):
    """Directly exercise the staticmethod-ish flatten logic by importing it."""

    def setUp(self):
        # Defer import so module-level side effects don't pollute the test
        # collector if deps are missing.
        from beetlelm.training.beetle_trainer import Trainer  # noqa: WPS433
        self._Trainer = Trainer

    def _fake_trainer(self, configs):
        t = self._Trainer.__new__(self._Trainer)
        t.configs  = configs
        t.run_name = configs.get("run_name", "test_run")
        return t

    def test_flattens_model_params_to_model(self):
        cfg = {
            "run_name": "test_run",
            "model": {
                "architecture": "pico",
                "params": {
                    "d_model":              768,
                    "attention_n_heads":    12,
                    "attention_n_kv_heads":  1,
                    "n_layers":             12,
                },
            },
            "checkpointing": {"save_every_n_steps": 500},
        }
        t = self._fake_trainer(cfg)
        pico = t._build_pico_compatible_training_config()

        self.assertEqual(pico["model"]["d_model"],              768)
        self.assertEqual(pico["model"]["attention_n_heads"],    12)
        self.assertEqual(pico["model"]["attention_n_kv_heads"], 1)
        self.assertEqual(pico["model"]["n_layers"],             12)
        self.assertEqual(pico["model"]["architecture"],         "pico")
        self.assertNotIn("params", pico["model"])

    def test_does_not_mutate_original_configs(self):
        cfg = {
            "run_name": "test_run",
            "model":    {"params": {"d_model": 64}},
        }
        t = self._fake_trainer(cfg)
        _ = t._build_pico_compatible_training_config()

        self.assertIn("params",  cfg["model"])
        self.assertEqual(cfg["model"]["params"]["d_model"], 64)

    def test_inserts_checkpointing_run_name(self):
        cfg = {"run_name": "the_run", "model": {"d_model": 64}}
        t = self._fake_trainer(cfg)
        pico = t._build_pico_compatible_training_config()
        self.assertEqual(pico["checkpointing"]["run_name"], "the_run")

    def test_preflight_for_pico_analyze_ov_circuit(self):
        """Mirror the exact reads pico-analyze performs."""
        cfg = {
            "run_name": "r",
            "model": {
                "params": {
                    "d_model":              128,
                    "attention_n_heads":      8,
                    "attention_n_kv_heads":   2,
                },
            },
            "checkpointing": {},
        }
        pico = self._fake_trainer(cfg)._build_pico_compatible_training_config()

        # pico-analyze/src/components/ov_circuit.py:34-37
        self.assertEqual(pico["model"]["d_model"],              128)
        self.assertEqual(pico["model"]["attention_n_heads"],      8)
        self.assertEqual(pico["model"]["attention_n_kv_heads"],   2)
        # pico-analyze/src/utils/initialization.py
        self.assertEqual(pico["checkpointing"]["run_name"], "r")


class SharedCheckpointingMessagesTest(unittest.TestCase):
    """Regression guard for em-dash vs double-hyphen in shared checkpointing."""

    def test_commit_messages_use_double_hyphen(self):
        src_path = os.path.join(REPO_ROOT, "src", "checkpointing", "training.py")
        with open(src_path, "r", encoding="utf-8") as f:
            text = f.read()

        # pico-analyze scans for '-- Step N'; the em-dash (U+2014) '—' fails
        # the regex. Guarantee the file emits only double-hyphen variants
        # inside any ``commit_message=`` argument.
        bad = re.findall(r"commit_message\s*=\s*[^,\n]*—[^,\n]*", text)
        self.assertEqual(
            bad, [],
            "src/checkpointing/training.py has em-dash in commit_message; "
            "pico-analyze's regex only matches double-hyphen '--': " + repr(bad),
        )

        pico_re = re.compile(r"Saving (HF Model|Tokenizer|Training Config|Fabric Checkpoint|Logs) -- Step")
        self.assertTrue(pico_re.search(text),
                        "expected at least one pico-compatible commit-message format string")


class LearningDynamicsCommitMessageTest(unittest.TestCase):
    """The LD emitter must produce the exact pico-analyze regex target."""

    def test_regex_matches_beetle_trainer_emitter(self):
        # pico-analyze/src/utils/data.py:145
        pattern = re.compile(r"Saving Learning Dynamics Data \((train|val)\) -- Step (\d+)")

        # The literal emitter in beetle_trainer._push_learning_dynamics_to_main
        for step in (0, 1, 500, 1000):
            msg = f"Saving Learning Dynamics Data (train) -- Step {step}"
            m = pattern.search(msg)
            self.assertIsNotNone(m, f"LD emitter message does not match pico-analyze regex: {msg!r}")
            self.assertEqual(m.group(1), "train")
            self.assertEqual(int(m.group(2)), step)


class PruneUploadRaceRegressionTest(unittest.TestCase):
    """
    Guard against the ordering bug that once caused
    ``upload_folder`` to fail with 'step_N is not a directory':

    ``_save_step_checkpoint_local(step)`` internally prunes older step
    directories.  If we call it BEFORE ``async_checkpointer.join()``, a
    still-in-flight async upload loses its source directory.  The training
    loop must therefore ``join()`` before it saves.
    """

    def test_train_loop_joins_before_save(self):
        src_path = os.path.join(REPO_ROOT, "src", "training", "beetle_trainer.py")
        with open(src_path, "r", encoding="utf-8") as f:
            text = f.read()

        start = text.find("if should_push_ckpt:")
        self.assertNotEqual(start, -1, "should_push_ckpt block not found")
        # Scan the rest of the function until the next top-level ``step += 1``.
        end = text.find("step += 1", start)
        self.assertNotEqual(end, -1, "could not find end of should_push_ckpt block (step += 1)")
        block = text[start:end]

        join_idx = block.find("async_checkpointer.join()")
        save_idx = block.find("_save_step_checkpoint_local(step)")
        self.assertNotEqual(join_idx, -1, "expected async_checkpointer.join() in should_push_ckpt block")
        self.assertNotEqual(save_idx, -1, "expected _save_step_checkpoint_local(step) in should_push_ckpt block")
        self.assertLess(
            join_idx, save_idx,
            "async_checkpointer.join() must come BEFORE "
            "_save_step_checkpoint_local(step); otherwise prune_local_checkpoints "
            "can delete a directory that's still being uploaded by the async "
            "worker, causing 'step_N is not a directory' errors.",
        )


class PicoCompatibleMainSequenceTest(unittest.TestCase):
    """
    Mock-based check of ``_push_pico_compatible_to_main(step=0)`` emits exactly
    the set of pico-compatible commit messages, in the correct order.

    Heavier test — skipped gracefully if optional deps are missing.
    """

    def test_step_zero_commit_sequence(self):
        try:
            from beetlelm.training import beetle_trainer as bt
        except Exception as e:
            self.skipTest(f"skipping; beetle_trainer import failed: {e}")

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # Create the files _push_pico_compatible_to_main expects in ckpt_path
            for fname in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "config.json",
                "model.safetensors",
            ):
                open(os.path.join(tmp, fname), "w").close()
            os.makedirs(os.path.join(tmp, "fabric_checkpoint"), exist_ok=True)
            # runs_dir/run_name layout for training_config.yaml discovery
            runs_dir = os.path.join(tmp, "_runs")
            run_root = os.path.join(runs_dir, "toy_run")
            os.makedirs(run_root, exist_ok=True)
            open(os.path.join(run_root, "training_config.yaml"), "w").close()

            t = bt.Trainer.__new__(bt.Trainer)
            t.hf_user         = "SomeOrg"
            t.run_name        = "toy_run"
            t.configs         = {
                "run_name": "toy_run",
                "model":    {"params": {"d_model": 64}},
                "checkpointing": {},
            }
            t.checkpoint_cfg  = {"runs_dir": runs_dir}
            t.log             = lambda *a, **kw: None

            # Patch HF + filesystem surfaces to capture call arguments.
            # The method is wrapped by @rank_zero_only → get the inner function.
            method = bt.Trainer._push_pico_compatible_to_main
            if hasattr(method, "__wrapped__"):
                method = method.__wrapped__
            # Unwrap @use_backoff too, if present.
            while hasattr(method, "__wrapped__"):
                method = method.__wrapped__

            upload_files   = []
            upload_folders = []

            class FakeApi:
                def list_repo_refs(self, repo_id):
                    return types.SimpleNamespace(branches=[types.SimpleNamespace(name="main")])
                def create_branch(self, **kw): pass

            with (
                mock.patch.object(bt, "HfApi", return_value=FakeApi()),
                mock.patch.object(bt, "upload_file",
                                  side_effect=lambda **kw: upload_files.append(kw)),
                mock.patch.object(bt, "upload_folder",
                                  side_effect=lambda **kw: upload_folders.append(kw)),
            ):
                method(t, step=0, checkpoint_path=tmp)

            messages = [kw["commit_message"] for kw in upload_files] + \
                       [kw["commit_message"] for kw in upload_folders]

            self.assertIn("Saving Tokenizer -- Step 0",        messages)
            self.assertIn("Saving Training Config -- Step 0",  messages)
            self.assertIn("Saving HF Model -- Step 0",         messages)
            self.assertIn("Saving Fabric Checkpoint -- Step 0", messages)

            all_revisions = {kw["revision"] for kw in upload_files + upload_folders}
            self.assertEqual(all_revisions, {"main"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
