# Dùng Native Thinking Tokens và Cấm Unicode Emoji

Status: accepted

Thay vì thêm token mới hoặc dạy mô hình định dạng thẻ XML `<thinking>...</thinking>` dễ xung đột với parser chuẩn, chúng ta giữ nguyên bộ native thinking/channel tokens có sẵn của Gemma 4 (`98: <|think|>`, `100: <|channel>`, `101: <channel|>`) để biểu diễn luồng suy luận, đồng thời đưa toàn bộ Unicode Emoji, Pictographs, Dingbats, ZWJ (`U+200D`) và Variation Selectors (`U+FE0F`) vào blacklist ở Bước 0 và cấm xuất hiện trong dữ liệu chưng cất Bước 1–2. Các biểu tượng cảm xúc ASCII đơn giản (`:)`, `:)))`, `._.`, `.-.`, `=))`) được phép xuất hiện có kiểm soát trong phần chat tự nhiên bằng dấu câu ASCII sẵn có, còn việc hiển thị icon đồ họa do tầng giao diện của `nuclear-code` đảm nhiệm.

## Considered Options

- **Ép cấu trúc `<thinking>...</thinking>` tùy biến**: Tăng nguy cơ học lại format từ đầu trên dữ liệu synthetic và tốn thêm token khi mã hóa thẻ XML.
- **Giữ lại ~2k–3k token Unicode Emoji**: Gây lãng phí slot trong cả `embed_tokens` lẫn `embed_tokens_per_layer` trên toàn bộ 35 tầng cho nhu cầu không thuộc về trợ lý code terminal.
- **Thêm custom token cho ASCII emoticon**: Phá vỡ tính đồng nhất vector nhúng tiền huấn luyện và hoàn toàn không cần thiết vì ký tự dấu câu ASCII đã đủ biểu diễn.

## Consequences

- Không phát sinh bất kỳ token mới nào trong bộ từ điển; tương thích thẳng với chat template gốc của Gemma 4 và parser suy luận của `llama.cpp`.
- Giải phóng thêm khoảng 2,000–3,000 slot trong bảng nhúng và PLE; nếu người dùng nhập Unicode emoji vào prompt thực tế, tokenizer vẫn xử lý an toàn qua 256 byte-fallback tokens (`<0x00>..<0xFF>`) mà không bị lỗi `<unk>`.
- Pipeline sinh dữ liệu Bước 1–2 phải áp luật cứng cấm 100% Unicode emoji trên toàn bộ mẫu và cấm ASCII emoticon bên trong kênh suy luận, code block và tool-call JSON.
