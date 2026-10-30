"""INT8/INT4 weight-only controls for D1."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quantization-pluse-mse" / "src"))

import torch

from codec.checkpoint import load_model
from pulsequant.core import apply_quant, assert_handle
from pulsequant.eval import decode_fixed, output_delta


class TestInt8Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = load_model("checkpoints/pulse-mse", "cpu", entropy=False).eval()
        cls.y_hat = torch.randn(1, 320, 12, 12)
        cls.before = decode_fixed(cls.model, cls.y_hat, 3)

    def test_int8_changes_and_restores(self):
        handle = apply_quant(self.model, "D1", qmax=127)
        self.assertTrue(handle.weights_changed)
        handle.restore()
        assert_handle(handle)

    def test_identity_does_not_change(self):
        handle = apply_quant(self.model, "D1", qmax=127, control="identity")
        self.assertFalse(handle.weights_changed)
        handle.restore()
        assert_handle(handle)

    def test_int8_changes_output(self):
        handle = apply_quant(self.model, "D1", qmax=127)
        after = decode_fixed(self.model, self.y_hat, 3)
        handle.restore()
        assert_handle(handle)
        self.assertGreater(output_delta(self.before, after), 0.0)

    def test_int4_hurts_more_than_int8(self):
        handle8 = apply_quant(self.model, "D1", qmax=127)
        after8 = decode_fixed(self.model, self.y_hat, 3)
        handle8.restore()
        assert_handle(handle8)

        handle4 = apply_quant(self.model, "D1", qmax=7, control="int4")
        after4 = decode_fixed(self.model, self.y_hat, 3)
        handle4.restore()
        assert_handle(handle4)

        delta8 = output_delta(self.before, after8)
        delta4 = output_delta(self.before, after4)
        self.assertGreater(delta4, delta8)


if __name__ == "__main__":
    unittest.main()
