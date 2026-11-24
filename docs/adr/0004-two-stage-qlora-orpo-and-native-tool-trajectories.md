# Hai giai đoạn QLoRA SFT -> ORPO, Custom Gemma 4 Serializer và Đóng băng Multimodal

Status: accepted

Qua kiểm tra trực tiếp `chat_template.jinja` của checkpoint chuẩn trên Hugging Face (`haku-nekooo/gemma4-e2b-nuclear-pruned` / `google/gemma-4-E2B-it`; không dùng bản GGUF cũ đã deprecated làm chuẩn), hàm `strip_thinking()` tự động xóa sạch khối `<|channel>thought...<channel|>` trong mọi lượt `model` khi gọi `apply_chat_template()`. Đồng thời, qua đọc mã nguồn `nc-agent` (`D:\Coding\nuclear_code`), `nc-agent` gọi `llama-server --jinja` qua `/v1/chat/completions` kèm GBNF grammar (`tool_call_gbnf()`, `planner_gbnf()`, `final_only_gbnf()`) nên model phải xuất `<|channel>thought\n...<channel|>` (được tách vào `reasoning_content`) theo sau bởi JSON envelope chuẩn xác (`{"thought":...,"action":{"tool":...,"args":...}}` hoặc `{"thought":...,"final":...}`). Vì vậy, ở Bước 1–2 chúng ta sử dụng bộ sinh `gemini-2.5-pro` (quỹ đạo agentic & code phức tạp/ORPO) kết hợp `gemini-2.5-flash` (giải thích & chat) qua Gemini Batch API, dùng **bộ serializer huấn luyện riêng cho Gemma 4** kèm assistant-only loss masking, rồi huấn luyện hai giai đoạn trên Colab Pro (L4/A100): Stage 2A chạy QLoRA SFT (train thêm `embed_tokens`, `lm_head` và `embed_tokens_per_layer` PLE ở learning rate thấp `1.5e-5` trong khi đóng băng hoàn toàn Vision/Audio encoder), theo sau bởi Stage 2B chạy một vòng ORPO ngắn (chỉ mở LoRA attention/MLP, đóng băng lại toàn bộ embedding và PLE) trên các cặp lỗi thực chiến của `nc-agent`.

## Considered Options

- **Dùng trực tiếp `tokenizer.apply_chat_template()` để render dữ liệu SFT**: Làm mất toàn bộ nhãn suy luận `<|channel>thought...<channel|>` ở lượt `model`.
- **Chỉ dạy native `<|tool_call>call:...<tool_call|>` mà bỏ qua GBNF envelope của `nc-agent`**: Không khớp với `tool_call_gbnf()` đang ép đầu ra của `llama-server` trong `nc-agent`, không trị được lỗi bịa tham số `edit_file` (`start_line`/`content`) hay lồng JSON vào `"final"`.
- **Chỉ chạy SFT thuần túy không có ORPO**: Dạy tốt format nhưng khó triệt để các lỗi rò/lồng JSON, đổi kiểu dữ liệu ngầm (`i32` sang `f64`) và lặp vòng ở model nhỏ.
- **Mở train cả embedding + PLE trong giai đoạn ORPO hoặc rã băng Vision/Audio**: Tăng mạnh VRAM trên Colab L4 24GB và có nguy cơ làm lệch trọng số cầu nối `mmproj` (`0.919 GB`) đã xuất ở Bước 0.

## Consequences

- Mọi kiểm chứng `chat_template.jinja` và parity test của bộ serializer phải đối chiếu trực tiếp với checkpoint Hugging Face (`haku-nekooo/gemma4-e2b-nuclear-pruned` / `google/gemma-4-E2B-it`).
- Bắt buộc băm SHA-256 toàn bộ tensor Vision và Audio encoder trước và sau khi merge adapter để bảo đảm `mmproj` không bị sai lệch sau huấn luyện.
- Pipeline Bước 1 tạo cả tập SFT (~3,500–5,000 mẫu) lẫn tập Preference Pairs cho ORPO (~500–800 cặp) bám sát schema thật của `nc-agent`.
