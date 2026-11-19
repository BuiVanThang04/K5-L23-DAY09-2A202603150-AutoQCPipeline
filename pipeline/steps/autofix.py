"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, TypedDict

import yaml

from ..common import ERROR, ROOT, Dataset, Issue, parse_yolo_line


DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


class Fix(TypedDict):
    """An edit recorded in the pipeline audit log."""

    image: str
    line: Optional[int]
    action: str
    before: str
    after: str


def _clip_box(text: str) -> Optional[str]:
    """Return a clipped YOLO row, or None if the box has no area."""
    class_id, cx, cy, width, height = parse_yolo_line(text)
    left, right = max(0.0, cx - width / 2), min(1.0, cx + width / 2)
    top, bottom = max(0.0, cy - height / 2), min(1.0, cy + height / 2)
    if right <= left or bottom <= top:
        return None
    clipped = (
        (left + right) / 2,
        (top + bottom) / 2,
        right - left,
        bottom - top,
    )
    return f"{class_id} " + " ".join(str(value) for value in clipped)


def fix(
    ds: Dataset, issues: Iterable[Issue], cfg: Dict[str, Any]
) -> List[Fix]:
    """Apply supported errors to the working copy and log each change."""
    if ds.root.resolve().is_relative_to((ROOT / "data" / "raw").resolve()):
        raise ValueError("Autofix requires a working copy of the raw dataset")

    fixes: List[Fix] = []
    errors: Dict[str, List[Issue]] = defaultdict(list)
    for issue in issues:
        if issue.severity == ERROR:
            errors[issue.image].append(issue)

    for stem, image_errors in sorted(errors.items()):
        codes = {issue.code for issue in image_errors}
        if codes & REMOVE_CODES or any(
            code.startswith("DUP_IMAGE_") for code in codes
        ):
            image = ds.image_path(stem)
            label_path = ds.label_path(stem)
            if image is not None or label_path.exists():
                if image is not None:
                    image.unlink()
                label_path.unlink(missing_ok=True)
                fixes.append(Fix(
                    image=stem, line=None, action="remove_image",
                    before="ảnh hoặc nhãn còn trong lô", after="đã loại",
                ))
            continue

        if "BAD_CLASS_NAME" in codes:
            try:
                before = ds.yaml_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                before = ""
            metadata = yaml.safe_load(before) or {}
            names = cfg["classes"]
            if metadata.get("names") != names:
                metadata["names"] = names
                after = yaml.safe_dump(metadata, allow_unicode=True)
                ds.yaml_path.write_text(after, encoding="utf-8")
                fixes.append(Fix(
                    image="data.yaml", line=None, action="rename_classes",
                    before=before, after=after,
                ))

        label_path = ds.label_path(stem)
        if not label_path.exists():
            if "MISSING_LABEL_FILE" in codes:
                fixes.append(Fix(
                    image=stem, line=None, action="needs_label",
                    before="(không có file)", after="chuyển người gán",
                ))
            continue

        line_codes: Dict[int, Set[str]] = defaultdict(set)
        for issue in image_errors:
            if issue.line is not None:
                line_codes[issue.line].add(issue.code)
        if not line_codes:
            continue

        lines = label_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        updated: List[str] = []
        n_fixes = len(fixes)
        for number, text in enumerate(lines, 1):
            row_codes = line_codes.get(number, set())
            if row_codes & DROP_CODES:
                replacement = None
            elif "BOX_OUT_OF_IMAGE" in row_codes:
                replacement = _clip_box(text)
            else:
                replacement = text

            if replacement is not None:
                updated.append(replacement)
            if replacement != text:
                fixes.append(Fix(
                    image=stem, line=number,
                    action="drop_line" if replacement is None else "clip_box",
                    before=text, after=replacement or "",
                ))
        if len(fixes) != n_fixes:
            content = "\n".join(updated) + ("\n" if updated else "")
            label_path.write_text(content, encoding="utf-8")

    return fixes
