from __future__ import annotations

from pathlib import Path

from brain_tumor_hpsc.data import summarize_image_dataset


def test_summarize_image_dataset(tmp_path: Path) -> None:
    for class_name in ["glioma", "notumor"]:
        (tmp_path / class_name).mkdir()

    (tmp_path / "glioma" / "sample.jpg").write_text("placeholder", encoding="utf-8")
    (tmp_path / "glioma" / "notes.txt").write_text("not an image", encoding="utf-8")

    summary = summarize_image_dataset(tmp_path, ["glioma", "notumor"])

    assert summary == {"glioma": 1, "notumor": 0}
