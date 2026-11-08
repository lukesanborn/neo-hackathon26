"""Pin the paper's pretraining schedule: Cell-JEPA App. E.1 trains on Kidney for 4 epochs, batch 32."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module_constants(path: Path) -> dict:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def test_pretrain_paper_epochs():
    from src.track_a.pretrain import PAPER_EPOCHS, PretrainConfig
    assert PAPER_EPOCHS == 4
    assert PretrainConfig(objective="jepa").epochs == PretrainConfig(objective="scgpt").epochs == 4


def test_modal_track_a_paper_schedule():
    c = _module_constants(ROOT / "modal_track_a.py")  # `import modal` may be unavailable
    assert c["PAPER_EPOCHS"] == 4
    assert c["PAPER_BATCH"] == 32
