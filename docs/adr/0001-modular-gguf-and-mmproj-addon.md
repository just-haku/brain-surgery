# Kiến trúc Modular GGUF (Core Brain + Multimodal Add-on)

Status: accepted

Hệ thống offline `nuclear-code` vận hành trên GPU GTX 1650 Ti (4GB VRAM), nơi việc nạp sẵn bộ mã hóa Vision và Audio vào cùng file model chính sẽ lãng phí VRAM trong các phiên lập trình thuần văn bản. Chúng ta quyết định giữ toàn bộ cầu nối đa phương thức trong quá trình phẫu thuật từ điển (`5: [multimodal]`, `255999..256000`, `258880..258884`), đóng băng Vision và Audio encoder khi chạy QLoRA trên Colab T4, và xuất tách thành hai tệp độc lập: `model-text-code.gguf` (Core Brain nạp mặc định) và `mmproj-vision-audio.gguf` (Multimodal Add-on chỉ mount qua `--mmproj` khi người dùng gọi tính năng ảnh hoặc âm thanh).

## Considered Options

- **Cắt bỏ vĩnh viễn Vision và Audio encoder**: Tiết kiệm công sức đồng bộ token nhưng làm mất hoàn toàn khả năng đọc ảnh chụp lỗi hoặc âm thanh offline về sau.
- **Đóng gói tất cả vào một mô hình luôn nạp sẵn đa phương thức**: Đơn giản hóa khâu nạp file nhưng chiếm thêm bộ nhớ VRAM cố định ngay cả khi chỉ chat và viết code.
- **Modular Add-on (`model-text-code.gguf` + `mmproj-vision-audio.gguf`)**: Chạy text/code siêu nhẹ out-of-the-box, vẫn gắn thêm đa phương thức theo nhu cầu.

## Consequences

- Các token đa phương thức gốc nằm ở cuối bảng (`255999..258884`) sẽ dịch sang dải ID mới sau khi cắt từ điển, buộc phải remap đồng bộ trong `config.json` (cả root lẫn `text_config`) và `tokenizer_config.json`.
- Phải chạy dry-run `convert_hf_to_gguf.py` cho cả model chính và `--mmproj` ngay cuối Bước 0 để phát hiện sớm lỗi tương thích. Nếu `llama.cpp` tại thời điểm xuất gặp lỗi khi đóng gói Audio vào `mmproj`, xuất trước Vision `mmproj` và giữ nguyên Audio encoder trong checkpoint `safetensors`.
