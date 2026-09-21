"""Run with the worker Torch environment: python tests/checkpoint_cuda_smoke.py.

Checks frozen inference against the trusted training definitions without running
training. This test extracts definitions from operator-controlled source only;
the production evaluator never reads or executes candidate source.
"""
import ast
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autolab/evaluator"))
from frozen_model import GPT, GPTConfig, fa3


class CheckpointCudaSmoke(unittest.TestCase):
    def test_round_trip_and_training_parity(self):
        import dataclasses
        import torch.nn as nn
        import torch.nn.functional as F
        tree = ast.parse((ROOT / "train.py").read_text())
        names = {"GPTConfig", "norm", "has_ve", "apply_rotary_emb", "CausalSelfAttention", "MLP", "Block", "GPT"}
        definitions = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names], type_ignores=[])
        namespace = {"torch": torch, "nn": nn, "F": F, "dataclass": dataclasses.dataclass, "fa3": fa3}
        exec(compile(definitions, "trusted_training_definitions", "exec"), namespace)
        torch.manual_seed(123)
        config = GPTConfig(n_layer=4, n_head=2, n_kv_head=2, n_embd=256, vocab_size=8192)
        original_config = asdict(config)
        original_config.pop("value_embedding_gates")
        original = namespace["GPT"](namespace["GPTConfig"](**original_config)).cuda()
        original.init_weights()
        # Exercise nonzero residual projections and gates, not just the neutral init.
        with torch.no_grad():
            for name, parameter in original.named_parameters():
                if "c_proj" in name or "ve_gate" in name:
                    parameter.normal_(0, 0.02)
        original.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({k: v.detach().cpu() for k, v in original.state_dict().items()}, path)
            weights = torch.load(path, map_location="cpu", weights_only=True)
        restored = GPT(config)
        restored.load_state_dict(weights, strict=True, assign=True)
        restored.cuda().eval()
        restored.cos, restored.sin = restored._precompute_rotary_embeddings(restored.rotary_seq_len, 128)
        inputs = torch.randint(0, 8192, (2, 256), device="cuda")
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            expected = original(inputs)
            actual = restored(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(restored.transformer.wte.weight.dtype, torch.bfloat16)
        broken = dict(weights)
        broken.pop("lm_head.weight")
        with self.assertRaises(RuntimeError):
            GPT(config).load_state_dict(broken, strict=True, assign=True)
        # Both declared variants must load strictly and execute finite inference.
        ungated_config = GPTConfig(**{**asdict(config), "value_embedding_gates": False})
        ungated = GPT(ungated_config)
        ungated.load_state_dict({k: v for k, v in weights.items() if "ve_gate" not in k}, strict=True, assign=True)
        ungated.cuda().eval()
        ungated.cos, ungated.sin = ungated._precompute_rotary_embeddings(ungated.rotary_seq_len, 128)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            self.assertTrue(torch.isfinite(ungated(inputs)).all().item())


if __name__ == "__main__":
    unittest.main()
