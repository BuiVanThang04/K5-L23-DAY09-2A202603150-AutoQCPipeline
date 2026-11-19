"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from typing import AbstractSet, Any, Dict, Iterable, List

from ..common import ERROR, WARN, Issue


def decide(
    issues: Iterable[Issue],
    image_stems: Iterable[str],
    cfg: Dict[str, Any],
    reviewed: AbstractSet[str] = frozenset(),
) -> Dict[str, str | int | float | List[str]]:
    """Decide whether the dataset can be released."""
    stems = set(image_stems)
    warning_images: set[str] = set()
    n_errors = 0
    for issue in issues:
        if issue.severity == ERROR:
            n_errors += 1
        elif (
            issue.severity == WARN
            and issue.image in stems
            and issue.image not in reviewed
        ):
            warning_images.add(issue.image)

    n_images = len(stems)
    n_warn_images = len(warning_images)
    ratio = n_warn_images / n_images if n_images else 0.0
    threshold = cfg["gate"]["warn_ratio_hold"]
    warning_summary = (
        f"{n_warn_images}/{n_images} ảnh còn cảnh báo chưa duyệt "
        f"({ratio:.2%}, ngưỡng Hold {threshold:.2%})"
    )
    if n_errors:
        decision = "Rework"
        reasons = [f"Còn {n_errors} lỗi bắt buộc trên lô {n_images} ảnh"]
    elif ratio >= threshold:
        decision = "Hold"
        reasons = [warning_summary]
    else:
        decision = "Release"
        reasons = [f"0 lỗi bắt buộc, {warning_summary}"]

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": round(ratio, 4),
        "reasons": reasons,
    }
