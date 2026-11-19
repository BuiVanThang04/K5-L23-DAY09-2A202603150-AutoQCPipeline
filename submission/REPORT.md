# Day 09 Lab — Bùi Văn Thắng: 2A202603150 - Bùi Văn Thắng

## 1. Kết luận trước

Hold (mã 10). Pipeline nhận 224 ảnh, loại 19 và xuất bản nháp gồm 205 ảnh. Sau khi duyệt mô phỏng 80 ảnh, còn 0 lỗi bắt buộc nhưng 38/205 ảnh (18,54%) vẫn có cảnh báo chưa duyệt, vượt ngưỡng 10%. [Tóm tắt lần chạy](reports/08_summary.json)

## 2. Lỗi và phân việc

Báo cáo vòng đầu có 204 dòng lỗi và cảnh báo, gồm 19 dòng ở intake và 185 dòng kiểm nhãn trên 205 ảnh còn lại. Một ảnh có thể xuất hiện ở nhiều dòng. [Danh sách lỗi vòng đầu](reports/02_error_list_v1.csv)

| Mã lỗi                                                                                 |        Số dòng | Ai xử lý và cách xử lý                                             |
| -------------------------------------------------------------------------------------- | -------------: | ------------------------------------------------------------------ |
| `BAD_SIZE`, `CORRUPT_IMAGE`                                                            |           3, 2 | Máy loại ảnh quá nhỏ hoặc không đọc được.                          |
| `DUP_IMAGE_EXACT`, `DUP_IMAGE_NEAR`, `LEAK_TEST`                                       |        8, 2, 4 | Máy loại ảnh trùng hoặc rò rỉ từ tập đánh giá.                     |
| `BAD_CLASS_ID`, `BAD_FORMAT`, `BOX_ZERO`, `DUP_BOX`                                    |     6, 3, 4, 7 | Máy xóa dòng lỗi. Người duyệt kiểm lại nếu việc xóa làm thiếu box. |
| `BOX_OUT_OF_IMAGE`                                                                     |              8 | Máy cắt box về trong ảnh.                                          |
| `BAD_CLASS_NAME`                                                                       |              2 | Máy sửa danh sách tên lớp trong YAML.                              |
| `MISSING_LABEL_FILE`                                                                   |              6 | Máy đưa vào hàng đợi, người gán nhãn.                              |
| `BOX_HUGE`, `BOX_TINY`, `CLASS_RARE`, `EMPTY_LABEL`                                    |    3, 6, 4, 12 | Người kiểm box, ảnh nhãn rỗng hoặc số mẫu của lớp.                 |
| `POSSIBLE_FALSE_POS`, `POSSIBLE_LOOSE_BOX`, `POSSIBLE_MISSING`, `POSSIBLE_WRONG_CLASS` | 25, 28, 53, 18 | Người đối chiếu ảnh để quyết định sửa nhãn.                        |

`img_0020` đứng đầu với 11,524 điểm vì vừa có dòng bị xóa, vừa có cảnh báo thiếu box và sai lớp. Trong 125 ảnh chờ duyệt, mô phỏng đã xử lý 80 ảnh, gồm 74 ảnh có ít nhất một cảnh báo `POSSIBLE_*` và 6 ảnh thiếu file nhãn. [Hàng đợi](reports/05_review_queue.csv) · [Log duyệt](reports/06_review_log.csv)

## 3. Calibration

| Tham số                 | Ngưỡng cũ → mới | Precision / recall trước → sau                                                                                 |
| ----------------------- | --------------- | -------------------------------------------------------------------------------------------------------------- |
| `duplicates.phash_dist` | 2 → 4           | Ảnh gần trùng: 1,00 / 0,50 → 1,00 / 0,67. Bắt thêm 1 ca trên 6 ca calib và giữ precision.                      |
| `second_opinion.conf`   | 0,25 → 0,60     | Thiếu box: 0,50 / 1,00 → 0,57 / 1,00. Sai lớp: 0,56 / 0,83 → 0,62 / 0,83. Box lỏng: 0,29 / 0,40 → 0,20 / 0,20. |

Tôi chọn `conf=0.60` để giảm báo nhầm về thiếu box và sai lớp, chấp nhận bỏ sót thêm một ca box lỏng trên calib. Với pHash, dù recall tăng trên calib, cả hai cặp gần trùng được báo trên raw đều có khoảng cách 0. Mức 2 cũng bắt được hai cặp này, nên mức tăng recall trên calib chưa thể hiện thành ca phát hiện thêm trên raw. [Calibration](calibration/calibration_report.md) · [Intake](reports/01_intake.txt)

## 4. Cửa chặn và điểm mù

Chạy riêng với `--budget 0` cho kết quả Rework (mã 20). Còn 6 lỗi `MISSING_LABEL_FILE` trên 205 ảnh và thư mục thử không có manifest xuất. Bằng chứng nằm trong `work/rep_nb/07_error_list_v3.txt`.

Trên calib ở ngưỡng đã chọn, pHash bỏ sót 2/6 ca gần trùng và bộ kiểm bỏ sót 4/5 ca box lỏng. Hai nhóm này vẫn cần người kiểm khi rà soát nhãn. [Kết quả sweep](calibration/calibration.json)

## 5. Train raw và clean

Tôi train YOLO26n trong 20 epoch với seed 0 và 1 cho mỗi biến thể, dùng cùng tập val và test.

| Bộ train | Số ảnh đầu vào |          mAP50 |      mAP50–95 |     Precision |        Recall |
| -------- | -------------: | -------------: | ------------: | ------------: | ------------: |
| Raw      |            224 | 0,726 ± 0,0002 | 0,512 ± 0,004 | 0,764 ± 0,061 | 0,649 ± 0,028 |
| Clean    |            205 |  0,700 ± 0,030 | 0,486 ± 0,013 | 0,715 ± 0,073 | 0,615 ± 0,017 |

Mỗi chỉ số là trung bình ± độ lệch chuẩn giữa hai seed. mAP50 của clean thấp hơn khoảng 0,025, nhỏ hơn mức dao động 0,030 giữa hai seed của clean. Với hai seed, tôi chưa đủ cơ sở kết luận bộ clean cho chất lượng mô hình thấp hơn. [Bảng kết quả](results/comparison.md)

Hai yếu tố cần kiểm thêm là 4 ảnh raw trùng test và 38/205 ảnh clean còn cảnh báo chưa duyệt. Có thể đánh giá lại bốn checkpoint trên cùng 96 ảnh test còn lại và kiểm thủ công một mẫu trong nhóm ảnh clean còn cảnh báo. [Phân tích riêng theo năm câu trong notebook](results/ANALYSIS.md)

## 6. Quyết định và ghi chú AI

Giữ Hold. Để phát hành lô 205 ảnh, cần xử lý và duyệt lại các ảnh còn cảnh báo, rồi chạy lại gate. Nếu số ảnh giữ nguyên, cần giảm từ 38 xuống tối đa 20 ảnh còn cảnh báo và giữ 0 lỗi bắt buộc.

- Yêu cầu chính: áp dụng bản nháp bốn module CP2 theo docstring và test có sẵn.
- Kiểm lại: chạy `pytest` được 79/79 và đối chiếu `DUP_BOX` ở dòng 3 của `img_0001` và dòng 2 của `img_0006`.
