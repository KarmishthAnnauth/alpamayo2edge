"""Adapter checkpoints carry the row-masked stage-1 tensors (D-046). Fake student."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill import checkpoint  # noqa: E402


class _Proj(nn.Module):
    def __init__(self, n_dom, d):
        super().__init__()
        self.fc = nn.Embedding(n_dom, d * d)
        self.bias = nn.Embedding(n_dom, d)


class _LM(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.layers = nn.Module()
        self.layers.lora_A = nn.Parameter(torch.randn(2, d))
        self.layers.lora_B = nn.Parameter(torch.randn(d, 2))

    def get_input_embeddings(self):
        return self.embed


class _Student:
    def __init__(self, old_n=6, new=4, d=3, n_dom=4, dom=2):
        self.lm = _LM(old_n + new, d)
        self.net = nn.Module()
        self.net.action2llm = _Proj(n_dom, d)
        self.net.llm2action = _Proj(n_dom, d)
        self.new_token_range = (old_n, old_n + new)
        self.action_domain_id = dom
        self.lora_stats = {"wrapped": 1}

    def _decoder_layers(self):
        return self.lm.layers


def _perturb(st, scale=10.0):
    with torch.no_grad():
        for p in [st.lm.layers.lora_A, st.lm.layers.lora_B, st.lm.embed.weight,
                  st.lm.lm_head.weight, st.net.action2llm.fc.weight, st.net.action2llm.bias.weight,
                  st.net.llm2action.fc.weight, st.net.llm2action.bias.weight]:
            p.add_(scale)


def test_rows_round_trip_and_pretrained_rows_untouched(tmp_path):
    torch.manual_seed(0)
    src = _Student()
    checkpoint.save_adapters(src, tmp_path, stage="stage1", epoch=3)
    dst = _Student()
    _perturb(dst)
    pre_embed = dst.lm.embed.weight.detach().clone()
    pre_fc = dst.net.action2llm.fc.weight.detach().clone()
    meta = checkpoint.load_adapters(dst, tmp_path)
    assert meta["epoch"] == 3
    old_n = src.new_token_range[0]
    # appended rows and our domain row restored ...
    assert torch.equal(dst.lm.embed.weight[old_n:], src.lm.embed.weight[old_n:])
    assert torch.equal(dst.lm.lm_head.weight[old_n:], src.lm.lm_head.weight[old_n:])
    assert torch.equal(dst.net.action2llm.fc.weight[2], src.net.action2llm.fc.weight[2])
    assert torch.equal(dst.lm.layers.lora_A, src.lm.layers.lora_A)
    # ... and nothing below the appended rows / on other domains moved
    assert torch.equal(dst.lm.embed.weight[:old_n], pre_embed[:old_n])
    assert torch.equal(dst.net.action2llm.fc.weight[[0, 1, 3]], pre_fc[[0, 1, 3]])


def test_geometry_mismatch_is_refused(tmp_path):
    src = _Student()
    checkpoint.save_adapters(src, tmp_path)
    dst = _Student(dom=1)
    with pytest.raises(RuntimeError, match="geometry"):
        checkpoint.load_adapters(dst, tmp_path)


def test_legacy_lora_only_file_still_loads(tmp_path, caplog):
    src = _Student()
    root = src._decoder_layers()
    torch.save({"adapters": {n: p.detach() for n, p in root.named_parameters()},
                "meta": {"lora": {}}}, tmp_path / checkpoint.ADAPTERS_NAME)
    dst = _Student()
    with caplog.at_level("WARNING"):
        checkpoint.load_adapters(dst, tmp_path)
    assert "NO stage-1 rows" in caplog.text
