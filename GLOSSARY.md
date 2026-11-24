# Brain Surgery

Ngữ cảnh thiết kế và tối ưu hóa mô hình ngôn ngữ cỡ nhỏ phục vụ suy luận lập trình song ngữ Anh–Việt trên phần cứng giới hạn bộ nhớ cho ứng dụng `nuclear-code`.

## Language

**Source Checkpoint**:
Bản trọng số gốc chưa lượng tử hóa được lưu trữ từ xa và dùng làm đầu vào duy nhất cho phẫu thuật từ điển và tinh chỉnh mô hình.
_Avoid_: Quantized baseline, raw weights

**Baseline Brain**:
Mô hình đã lượng tử hóa được giữ sẵn ở máy cá nhân chỉ nhằm đo chuẩn chất lượng đầu ra và mức tiêu thụ bộ nhớ trước khi phẫu thuật.
_Avoid_: Training source, source model

**Core Brain**:
Mô hình xử lý văn bản và mã nguồn độc lập được `nuclear-code` nạp mặc định để giữ mức tiêu thụ bộ nhớ thấp nhất trong tác vụ lập trình thường ngày.
_Avoid_: Full model, base brain

**Multimodal Add-on**:
Mô-đun thị giác và âm thanh tách rời chỉ được gắn bổ sung vào Core Brain theo nhu cầu để không chiếm bộ nhớ khi chỉ làm việc với văn bản hoặc mã nguồn.
_Avoid_: Vision model, baked-in multimodal

**Vocab Surgery**:
Quá trình cắt bỏ đồng bộ các mục từ thuộc ngôn ngữ dư thừa và biểu tượng cảm xúc khỏi bộ từ điển cùng toàn bộ các bảng nhúng đánh chỉ mục theo từ, đồng thời bảo toàn token điều khiển, byte dự phòng và cầu nối đa phương thức.
_Avoid_: Naive pruning, tokenizer trimming

**BPE Merge-Closure**:
Bất biến bắt buộc mọi token được giữ lại phải kéo theo đầy đủ các cặp token tổ tiên cấu thành nên nó trong cây ghép từ.
_Avoid_: Regex filtering, isolated token filtering

**Native Reasoning**:
Luồng suy luận từng bước nội tại được biểu diễn bằng kênh suy nghĩ có sẵn của mô hình thay vì gắn thêm thẻ đánh dấu tự chế.
_Avoid_: Custom XML thinking tags

**ASCII Emoticon Policy**:
Quy tắc loại bỏ hoàn toàn biểu tượng Unicode khỏi bộ từ điển và dữ liệu huấn luyện, chỉ cho phép biểu tượng mặt cười ASCII đơn giản như `:)`, `:)))`, `._.`, `.-.`, `=))` trong câu trò chuyện tự nhiên.
_Avoid_: Emoji support, custom emoticon tokens

**Agentic Trajectory**:
Chuỗi hội thoại nhiều lượt mô phỏng đầy đủ vòng suy luận, gọi công cụ, nhận kết quả và phản hồi cuối cùng của trợ lý lập trình.
_Avoid_: Single-turn prompt, chat log

**Preference Pair**:
Cặp mẫu đối chiếu gồm một phản hồi đạt chuẩn và một phản hồi mắc lỗi thực tế trên cùng ngữ cảnh, dùng để điều chỉnh hành vi mô hình tránh lặp lại sai sót.
_Avoid_: Positive-only example, negative prompt
