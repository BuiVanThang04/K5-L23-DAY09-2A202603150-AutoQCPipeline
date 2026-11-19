# Phân tích kết quả train Day 09

## 1. Chênh lệch mAP50 và độ dao động giữa các seed

| Bộ train | Seed 0 | Seed 1 | Trung bình ± độ lệch chuẩn |
|---|---:|---:|---:|
| Raw | 0,7256 | 0,7259 | 0,7258 ± 0,0002 |
| Clean | 0,6788 | 0,7219 | 0,7004 ± 0,0305 |

Raw đạt mAP50 cao hơn ở cả hai seed. Chênh lệch trung bình `clean − raw` là **−0,0254**, nhỏ hơn độ lệch chuẩn **0,0305** của clean. Mức chênh cũng giảm từ 0,0468 ở seed 0 xuống 0,0040 ở seed 1. Với hai seed, tôi chưa đủ cơ sở để kết luận clean cho kết quả kém hơn. [Bảng tổng hợp](comparison.md)

## 2. Số ảnh raw thực sự dùng khi train

Tôi kiểm tra `raw/train/labels.cache` trên Colab. Ultralytics giữ lại **213/224 ảnh raw** sau khi quét ảnh và nhãn, tức là **11 ảnh bị loại**. Số 224 trong JSON là số file ảnh đầu vào trước khi quét. [Kết quả raw seed 0](raw_seed0.json) · [Kết quả raw seed 1](raw_seed1.json) · [Script đếm ảnh](../../scripts/train_compare.py)

## 3. Ảnh raw trùng với tập test

Pipeline phát hiện **4 ảnh trong raw trùng với tập test**. Tôi chưa đo được chúng làm tăng mAP50 của raw bao nhiêu. Để kiểm tra, có thể đánh giá lại cả bốn checkpoint trên cùng **96 ảnh test còn lại** rồi so chênh lệch raw và clean với kết quả trên đủ 100 ảnh. [Báo cáo intake](../reports/01_intake.txt)

## 4. Kết quả theo lớp

| Lớp | AP50 raw | AP50 clean | Chênh lệch |
|---|---:|---:|---:|
| car | 0,7375 | 0,7485 | +0,0110 |
| person | 0,8221 | 0,8274 | +0,0053 |
| truck | 0,6485 | 0,5588 | −0,0897 |
| bus | 0,6949 | 0,6667 | −0,0283 |

`car` tăng nhiều nhất, khoảng **0,011 AP50**. Khi xem log sửa, tôi thấy box `car` ở dòng 2 của `img_0078` bị tràn và đã được cắt về trong ảnh. Đây là một ví dụ nhãn car được sửa. Cần kiểm thêm để biết việc sửa box đóng góp bao nhiêu vào mức tăng AP. [Lỗi vòng đầu](../reports/02_error_list_v1.csv) · [Log tự sửa](../reports/03_fixes.csv)

## 5. Phần còn sót trong clean

Hàng đợi có **125 ảnh**, trong khi chỉ duyệt được **80 ảnh**. Sau QC cuối, còn **0 lỗi bắt buộc** nhưng **38/205 ảnh (18,54%)** vẫn có cảnh báo chưa duyệt. Trong nhóm này có 16 cảnh báo box lỏng và 11 cảnh báo nghi thừa box. Tôi giữ kết luận **Hold**. Cần duyệt hết các ca còn lại trước khi dùng bộ nhãn này làm mốc so sánh cho dữ liệu đã kiểm đầy đủ. [Hàng đợi](../reports/05_review_queue.csv) · [QC vòng cuối](../reports/07_error_list_v3.csv) · [Kết luận](../reports/08_summary.json)
