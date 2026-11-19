"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from ..common import ERROR, WARN, Dataset, Issue, iou_xywh, parse_yolo_line


def run(ds: Dataset, cfg: Dict[str, Any]) -> List[Issue]:
    """Validate each YOLO label row while preserving its original line number."""
    settings = cfg["label_format"]
    issues: List[Issue] = []
    check = "label_format"

    for label_path in ds.label_files():
        stem = label_path.stem
        lines = ds.read_label_lines(stem) or []
        if not lines:
            continue

        image_size = ds.image_size(stem)
        previous: Dict[int, List[Tuple[float, float, float, float]]]
        previous = defaultdict(list)
        for line_number, text in lines:
            try:
                class_id, cx, cy, width, height = parse_yolo_line(text)
            except (ValueError, OverflowError) as exc:
                issues.append(Issue(
                    check, ERROR, "BAD_FORMAT", stem,
                    f"Dòng nhãn sai định dạng: {exc}", line_number,
                ))
                continue

            if not 0 <= class_id < len(ds.classes):
                issues.append(Issue(
                    check, ERROR, "BAD_CLASS_ID", stem,
                    f"ID lớp {class_id} ngoài danh sách", line_number,
                ))
                continue

            if width <= 0 or height <= 0:
                issues.append(Issue(
                    check, ERROR, "BOX_ZERO", stem,
                    "Chiều rộng và chiều cao phải dương", line_number,
                ))
                continue

            left, right = cx - width / 2, cx + width / 2
            top, bottom = cy - height / 2, cy + height / 2
            overflow = max(0.0, -left, -top, right - 1.0, bottom - 1.0)
            if overflow > settings["overflow_tol"]:
                issues.append(Issue(
                    check, ERROR, "BOX_OUT_OF_IMAGE", stem,
                    f"Box tràn mép ảnh {overflow:.6g}", line_number,
                    {"overflow": overflow},
                ))

            box = (cx, cy, width, height)
            boxes = previous[class_id]
            if any(
                iou_xywh(box, earlier) > settings["dup_box_iou"]
                for earlier in boxes
            ):
                issues.append(Issue(
                    check, ERROR, "DUP_BOX", stem,
                    "Box trùng với một dòng trước cùng lớp", line_number,
                ))
            boxes.append(box)

            if image_size is not None:
                image_width, image_height = image_size
                short_side = min(width * image_width, height * image_height)
                if short_side < settings["min_box_px"]:
                    issues.append(Issue(
                        check, WARN, "BOX_TINY", stem,
                        f"Cạnh ngắn của box là {short_side:.2f} px", line_number,
                    ))
            if width * height > settings["max_box_frac"]:
                issues.append(Issue(
                    check, WARN, "BOX_HUGE", stem,
                    f"Box chiếm {width * height:.1%} diện tích ảnh", line_number,
                ))

    return issues
