# Vì sao bộ 50k được sử dụng?

Nếu có một bộ dữ liệu Pháp–Anh 50k thực thụ thay vì bộ Việt–Anh hiện hữu, pipeline đã
sẵn sàng: chỉ cần cấu hình `DATA_DIR` tương ứng trong `run_all.py` và chạy lại toàn bộ
các split cùng seed.

## Quy mô và split

Repository có hai bộ song ngữ: `fr_en_train.txt` chứa 300,000 cặp Pháp–Anh và
`vi_en_train.txt` chứa đúng 50,000 cặp Việt–Anh. FE_AGAL không tự giả định một
split riêng cho FR–EN vì file được đánh tên `*_train.txt`. Kịch bản A lấy một
subset có tách bằng khóa câu nguồn từ hàng triệu? thực tế từ toàn bộ 300k Pháp–Anh.

Kích thước 50k đề cập trong requirement tương ứng đúng tập Việt–Anh trong dự án, nhưng
nhiệm vụ evaluation Pháp–Anh được gắn trực tiếp với 300k dòng `fr_en_train.txt`.
Không có metadata train/dev/test đi kèm file FR–EN, nên split mới được tạo bằng
seeding cố định và log trong `data_splits.py`.

## Sweet spot

Với 300k corpus hiện hữu, giữ toàn bộ cặp FR–EN cho training gần với cấu hình dự án
`fr_train_size: 300000`, còn 50k là kích thước rõ ràng của tập Việt–Anh. Vì vậy
chúng không đối thủ trực tiếp: 50k không phải FR data. Không có learning curve chạy
sẵn trong repository; đánh giá học 50k/100k chỉ có thể suy ra từ cấu hình huấn luyện
và chi phí, không phải tuyên bố thực nghiệm chưa có bằng chứng. Theo nguyên tắc
scaling, 50k cân bằng iteration, compute và thu thập dữ liệu; 10k dễ overfit và
100k/full tăng thời gian  với return giảm dần nếu coverage-domain tương đương.
