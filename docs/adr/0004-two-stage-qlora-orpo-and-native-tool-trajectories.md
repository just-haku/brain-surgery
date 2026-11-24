# Hai giai đoạn QLoRA SFT -> ORPO, Native Gemma 4 Tool Calling và Đóng băng Multimodal

Status: accepted

Qua kiểm tra trực tiếp `chat_template.jinja` của checkpoint chuẩn trên Hugging Face (`haku-nekooo/gemma4-e2b-nuclear-pruned` / `google/gemma-4-E2B-it`; không dùng bản GGUF cũ đã deprecated làm chuẩn), hàm `strip_thinking()` tự động xóa sạch khối `<|channel>thought...<channel|>` trong mọi lượt `model` khi gọi `apply_chat_template()` mà không kèm `reasoning_content` / `preserve_thinking`. Đồng thời, để `nc-agent` (`nuclear-code`) không bị bó cứng vào JSON envelope của GBNF mà tận dụng được chế độ **`NativeTools`** qua `llama-server --jinja` (gửi mảng `tools`, đọc `reasoning_content` và `tool_calls`), Bước 1 sinh trực tiếp các quỹ đạo **Native Gemma 4 Tool Calling** (`<|tool>declaration:...<tool|>`, `<|channel>thought\n...\n<channel|>`, `<|tool_call>call:...<tool_call|>`, `<|tool_response>response:...<tool_response|>`) bằng `gemini-2.5-pro` + `gemini-2.5-flash` qua Gemini Batch API, tuần tự hóa qua **bộ serializer riêng cho Gemma 4** kèm assistant-only loss masking (mask toàn bộ `<|tool_response>`), rồi huấn luyện hai giai đoạn trên Colab Pro (L4/A100): Stage 2A QLoRA SFT (kèm `embed_tokens`, `lm_head` và `embed_tokens_per_layer` PLE ở learning rate thấp `1.5e-5` với Vision/Audio encoder đóng băng) tiếp nối bằng Stage 2B ORPO ngắn (chỉ mở LoRA attention/MLP, đóng băng lại toàn bộ embedding và PLE).

## Considered Options

- **Dùng trực tiếp `tokenizer.apply_chat_template()` để render dữ liệu SFT**: Không trả về character spans cho assistant-only loss masking trên các lượt `<|tool_response>` lồng trong cùng lượt `model`.
- **Giữ khuôn GBNF JSON wrapper cho mọi mẫu `nc-agent`**: Gây suy nghĩ kép tốn token và xung đột với hướng nâng cấp `NativeTools` đa provider trong `nuclear-code` (`CHANGE-SUGGESTIONS.md`).
- **Chỉ chạy SFT thuần túy không có ORPO**: Khó loại triệt để lỗi đổi kiểu dữ liệu ngầm (`i32` sang `f64`), bịa tham số `edit_file` (`start_line`/`content`), rò JSON thô hoặc lặp vòng ở model 2B.
- **Mở train cả embedding + PLE trong giai đoạn ORPO hoặc rã băng Vision/Audio**: Tăng mạnh VRAM trên Colab L4 24GB và có nguy cơ làm lệch cầu nối `mmproj` (`0.919 GB`).

## Consequences

- `formatter.py` phải khớp từng ký tự với `chat_template.jinja` của `haku-nekooo/gemma4-e2b-nuclear-pruned` khi chạy kiểm chứng parity ở đầu Bước 1 (`step1_distill.ipynb`).
- `nuclear-code` nâng cấp `nc-agent` sang `ToolCallingMode::NativeTools` theo [`CHANGE-SUGGESTIONS.md`](../../../nuclear_code/CHANGE-SUGGESTIONS.md) để khai thác trực tiếp năng lực native tool calling + `--jinja` sau khi fine-tune.
- Phải băm SHA-256 toàn bộ tensor Vision và Audio encoder trước và sau khi merge adapter ở Bước 2 để bảo đảm `mmproj` giữ nguyên vẹn.
