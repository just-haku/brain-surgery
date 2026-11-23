# Cắt đồng bộ Vocab + PLE với BPE Merge-Closure

Status: accepted

Gemma 4 E2B có bộ từ điển `262,144` token và `514,906` quy tắc BPE merge, trong đó khoảng 2.8B trên tổng ~5.1B tham số nằm ở bảng nhúng `embed_tokens` (chia sẻ trọng số với `lm_head`) và bảng nhúng theo từng tầng `embed_tokens_per_layer` (PLE, 35 layers × 256 chiều × 262,144). Chúng ta quyết định cắt đồng bộ `embed_tokens`, `lm_head` (nếu tách rời), `embed_tokens_per_layer` và `tokenizer.json` theo phương pháp Hybrid (Whitelist bắt buộc + Blacklist `<unused...>` và Unicode Emoji + lọc script Latin/VI/Code + xếp hạng tần suất trên corpus EN/VI/Code + đệ quy `BPE Merge-Closure`) về kích thước mục tiêu `49,152–65,536` token (bội số của 128), đồng thời xóa hoặc rebuild dứt điểm `tokenizer.model` để toàn bộ công cụ chỉ đọc đúng bảng ID đã phẫu thuật trong `tokenizer.json`.

## Considered Options

- **Chỉ cắt `embed_tokens` và `lm_head`**: Làm gãy kích thước chỉ mục với `embed_tokens_per_layer` (~2.35B tham số) và bỏ lỡ phần giảm VRAM lớn nhất của Gemma 4 E2B.
- **Lọc thuần Unicode/Regex không dùng corpus**: Dễ cài đặt nhưng giữ lại toàn bộ từ vựng Latin của ~50 ngôn ngữ châu Âu (~95k–130k token), chỉ cắt được 50–60% bảng nhúng.
- **Cắt cực gọn ~32,768 token thuần corpus**: Tiết kiệm VRAM tối đa nhưng tăng rủi ro xé lẻ từ tiếng Việt hoặc mã nguồn hiếm thành chuỗi byte-fallback dài.
- **Hybrid + BPE Merge-Closure mục tiêu `49,152–65,536` token**: Loại bỏ rác đa ngữ châu Âu và emoji mà vẫn giữ đủ độ phủ EN, VI, Code và bảo toàn cây ghép BPE.

## Consequences

- Toàn bộ vector BF16 của các token sống sót được giữ nguyên giá trị gốc, cho phép kiểm chứng độ khớp logit (`max_abs_diff < 1e-3`) ngay ở Bước 0 trước khi tốn chi phí huấn luyện.
- Mọi token ID bị dịch chuyển bắt buộc phải remap nhất quán ở cả tầng root của `config.json`, object lồng `text_config`, `generation_config.json` và `tokenizer_config.json`, tuyệt đối không để tồn tại `tokenizer.model` cũ lệch chỉ mục.
