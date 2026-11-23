# Kế hoạch Bước 0: Vocabulary & PLE Pruning cho Gemma 4 E2B

## 1. Mục tiêu & Đầu vào / Đầu ra

- **Mục tiêu**: Cắt giảm bộ từ điển `262,144` token của Gemma 4 E2B xuống mục tiêu **`49,152–65,536` token** (bội số của `128`), chỉ giữ **Tiếng Anh + Tiếng Việt + Code + Special/Multimodal tokens + Byte-fallback + ASCII emoticons**, loại bỏ hoàn toàn ~180 ngôn ngữ thừa, `6,227` slot `<unused...>` và toàn bộ Unicode Emoji. Giảm mạnh dung lượng bảng `embed_tokens` và `embed_tokens_per_layer` (PLE) để chuẩn bị cho fine-tune QLoRA trên Colab T4 (15GB VRAM) và chạy mượt trên GTX 1650 Ti (4GB VRAM).
- **Đầu vào (Source Checkpoint)**: `treadon/gemma4-E2B-it-abliterated` (BF16 `safetensors`; dự phòng: `Austriani/Gemma-4-E2B-it-ARA-Abliterated`).
  - Lưu ý: Tệp local `brain/Gemma-4-E2B-IT-AB.gguf` (`HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive`, `Q4_K_M`, 3.45 GB, text-only) chỉ đóng vai trò **Baseline Brain** để đối chiếu chất lượng và VRAM, không dùng làm đầu vào phẫu thuật.
- **Đầu ra (Artifact đẩy lên Hugging Face Private Repo)**:
  - Thư mục checkpoint BF16 `safetensors` đã phẫu thuật (`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `processor_config.json` nếu có, các shard `*.safetensors` + `model.safetensors.index.json`).
  - `id_map.json`: ánh xạ đơn định `old_id -> new_id` và `kept_old_ids`.
  - `pruning_report.json`: thống kê số token theo từng tầng lọc, số lượng BPE merges trước/sau, dung lượng tensor giảm được và kết quả 4 cổng kiểm chứng.
- **Lưu ý tài nguyên Colab T4 (15GB VRAM / ~12.7GB System RAM)**: Bản BF16 đầy đủ (~5.1B tổng tham số, ~10.2 GB trọng số) rất sát ngưỡng System RAM miễn phí của Colab. Khi phẫu thuật trên CPU, ưu tiên xử lý từng shard `safetensors` bằng `safetensors.safe_open` hoặc `low_cpu_mem_usage=True`, giải phóng bộ nhớ triệt để (`del` + `gc.collect()`) trước bước kiểm chứng mô hình gốc vs mô hình đã cắt.

---

## 2. Bản đồ Kiến trúc & Bất biến Gemma 4 E2B

Dựa trên kiểm tra trực tiếp metadata kiến trúc Gemma 4 E2B:

- **Từ điển & BPE**: `vocab_size = 262,144`, `514,906` quy tắc BPE merge, `add_space_prefix = False`, `add_bos_token = False`.
- **Kiến trúc Decoder & PLE**: `35` layers, `embedding_length = 1536`, `embedding_length_per_layer_input = 256`, `final_logit_softcapping = 30.0`, hybrid attention 5:1 với SWA 512 và shared KV 20 layers.
- **Dynamic Tensor Audit (Bắt buộc quét động)**: Trước khi cắt, script phải quét toàn bộ khóa tensor trong `model.safetensors.index.json` (hoặc toàn bộ file `*.safetensors`) để liệt kê mọi tensor có kích thước một chiều bằng `262,144` nhằm không bỏ sót do khác tiền tố module (`model.language_model.*` vs `language_model.model.*`), tối thiểu gồm:
  - `...embed_tokens.weight` (`[262144, 1536]`, thường chia sẻ trọng số với `lm_head`).
  - `...embed_tokens_per_layer.weight` (`[262144, 8960]` tức `35 × 256` chiều — nguồn tiết kiệm VRAM lớn nhất, ~2.35B tham số).
  - `...lm_head.weight` (nếu `tie_word_embeddings` là `false` hoặc tách tensor riêng).
- **Các mô-đun giữ nguyên bất biến**: Vision encoder, Audio encoder, các multimodal projector, và toàn bộ 35 transformer blocks (`attention`, `mlp`, `norm`, `per_layer_model_projection`, v.v.) tuyệt đối không thay đổi trọng số ở Bước 0.

---

## 3. Quy trình Lọc Từ điển 4 Tầng (Hybrid + BPE Merge-Closure)

### Tầng 1 — Mandatory Whitelist (Giữ vô điều kiện)

1. **Token hệ thống & điều khiển cốt lõi (`0..5`)**:
   - `0`: `<pad>`, `1`: `<eos>`, `2`: `<bos>`, `3`: `<unk>`, `4`: `<mask>`.
   - `5`: `[multimodal]` — **cảnh báo quan trọng**: trong bảng token gốc, `ID 5` có `token_type=1` (normal) chứ không mang cờ control (`type=3`), bắt buộc ép whitelist theo ID/tên để không làm đứt cầu nối đa phương thức.
2. **Token Tool Calling (`46..52`)**:
   - `46`: `<|tool>`, `47`: `<tool|>`, `48`: `<|tool_call>`, `49`: `<tool_call|>`, `50`: `<|tool_response>`, `51`: `<tool_response|>`, `52`: `<|"|>`.
3. **Token Native Thinking & Chat Turn (`98, 100..101, 105..106`)**:
   - `98`: `<|think|>`, `100`: `<|channel>`, `101`: `<channel|>`, `105`: `<|turn>`, `106`: `<turn|>`.
4. **256 Byte-Fallback Tokens (`238..493`)**:
   - Toàn bộ `<0x00>` đến `<0xFF>` (`token_type=6`), bảo đảm tokenizer không bao giờ sinh `<unk>` khi gặp ký tự lạ hoặc Unicode emoji từ input người dùng.
5. **Token Cầu nối Multimodal (`255999..256000`, `258880..258884`)**:
   - `255999`: `<|image>`, `256000`: `<|audio>`, `258880`: `<|image|>`, `258881`: `<|audio|>`, `258882`: `<image|>`, `258883`: `<audio|>`, `258884`: `<|video|>`.
6. **Dynamic Config & Chat-Template Audit**:
   - Tự động đọc `config.json` (cả root và `text_config`), `generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens_decoder`, và `chat_template` (Jinja2) để tự động whitelist thêm mọi token ID hoặc chuỗi đặc biệt được tham chiếu (phòng trường hợp repo `treadon` có khác biệt nhỏ về index so với GGUF baseline).
7. **Token ASCII, Whitespace/Indent & ASCII Emoticons**:
   - Toàn bộ ký tự ASCII in được (`0x20..0x7E`, gồm cả phiên bản có tiền tố `▁` trong SentencePiece/BPE).
   - Các token khoảng trắng, tab và xuống dòng nhiều cấp phục vụ thụt lề code (`107..115`: `\n`..`\n\n\n\n\n\n\n\n\n`, các cụm `▁+` và `\t+`).
   - Toàn bộ dấu câu/toán tử code và các cụm dấu câu cấu thành ASCII emoticons (`:)`, `:)))`, `._.`, `.-.`, `=))`).

### Tầng 2 — Mandatory Blacklist (Loại bỏ trừ khi nằm trong Tầng 1)

1. **`<unused...>` tokens**: Loại bỏ toàn bộ `6,227` slot từ `ID 6` đến `262143` có dạng `<unused...>`.
2. **Unicode Emoji & ký tự điều biến biểu tượng**:
   - Chặn toàn bộ các khối Emoji, Pictographs, Dingbats, Miscellaneous Symbols, Regional Indicators, Skin-Tone Modifiers (`U+2600..U+26FF`, `U+2700..U+27BF`, `U+1F1E0..U+1F1FF`, `U+1F300..U+1F6FF`, `U+1F900..U+1F9FF`, `U+1FA70..U+1FAFF`, `U+1F3FB..U+1F3FF`).
   - Chặn Zero Width Joiner (`U+200D`) và Variation Selectors (`U+FE00..U+FE0F`, đặc biệt `U+FE0F`) để giải phóng thêm ~2,000–3,000 slot trong `embed_tokens` và `PLE`.
3. **Non-Latin Scripts**: Loại bỏ toàn bộ token chứa ký tự CJK, Cyrillic, Arabic, Devanagari, Thai, Hangul, Kana, Hebrew, v.v.

### Tầng 3 — Lọc Script & Xếp hạng Tần suất trên Corpus EN + VI + Code

1. **Lọc ứng viên sơ bộ (Script Filter)**:
   - Sau bước decode an toàn (bỏ qua byte-fallback và special tokens), chỉ giữ ứng viên thuộc bảng chữ cái ASCII + nguyên âm/phụ âm tiếng Việt có dấu (hợp chuẩn NFC/NFD) + ký tự Code/Markdown.
   - Loại bỏ rác từ vựng mang dấu đặc trưng của các ngôn ngữ châu Âu khác (`ß`, `ø`, `ł`, `ñ`, `ç`, `ğ`, `ş`, `ä`, `ü`..., lưu ý giữ chữ `đ`/`Đ` của tiếng Việt).
2. **Stream Corpus nhẹ trên Colab (không lưu dataset về repo Git)**:
   - **Tiếng Anh**: lát cắt mẫu `HuggingFaceFW/fineweb-edu`.
   - **Tiếng Việt**: `wikimedia/wikipedia` (`vi`) + lát cắt `HuggingFaceFW/fineweb-2` (`vie_Latn`).
   - **Code**: `bigcode/the-stack-smol` hoặc `codeparrot/github-code-clean` (tập trung Python, C/C++, JS/TS, HTML/CSS, SQL, Bash, JSON, Markdown).
3. **Xếp hạng & Cân bằng Quota**:
   - Đếm tần suất xuất hiện token theo tokenizer gốc trên 3 miền corpus; đặt **ngưỡng sàn (floor quota)** bắt buộc cho nhóm từ/âm tiết tiếng Việt có dấu và nhóm token Code (từ khóa, indent, ngoặc, chuỗi phổ biến) để lượng mẫu tiếng Anh không đẩy văng từ vựng tiếng Việt hoặc code hiếm.
   - Chọn ngưỡng ứng viên mục tiêu để sau bước merge-closure đạt **`49,152–65,536` token**.

### Tầng 4 — BPE Merge-Closure & Căn chỉnh Tensor Core

1. **Đóng cây BPE Merge (Merge-Closure)**:
   - Đọc danh sách `merges` trong `tokenizer.json` (kiểm tra định dạng thực tế `"A B"` hoặc `["A", "B"]`).
   - Lặp đệ quy tới điểm bất động: với mỗi token nằm trong tập giữ lại nếu được sinh bởi quy tắc `(A, B)`, bắt buộc đưa cả `A` và `B` vào tập giữ lại. Không cho phép tồn tại token "mồ côi" thiếu thành phần cha.
2. **Cắt & Bảo toàn Thứ tự BPE Merges**:
   - Chỉ giữ lại các quy tắc `(A, B)` mà cả `A`, `B`, và token đích `A+B` đều nằm trong tập giữ lại cuối cùng.
   - Bắt buộc giữ nguyên **thứ tự tương đối** của danh sách `merges` gốc để hành vi gộp subword không bị sai lệch thứ tự ưu tiên.
3. **Căn chỉnh Bội số 128 (`% 128 == 0`)**:
   - Bù thêm (hoặc tỉa từ rìa tần suất thấp nhất chưa có con phụ thuộc trong merge tree) để tổng `new_vocab_size` chia hết cho `128` (ví dụ `49,152` hoặc `65,536`) nhằm tối ưu hiệu suất nhân ma trận trên Tensor Core / CUDA kernel của `llama.cpp` và PyTorch.
4. **Đánh chỉ mục lại đơn định**:
   - Sắp xếp `kept_old_ids` theo thứ tự tăng dần để giữ nguyên vị trí đầu bảng của `0..5` (`<pad>`, `<eos>`, `<bos>`, `<unk>`, `<mask>`, `[multimodal]`) và sinh `id_map.json` (`old_id -> new_id`).

---

## 4. Checklist Phẫu thuật Weights, Config & Tokenizer

Thực hiện theo đúng thứ tự dưới đây:

- [ ] **1. Audit & Slice toàn bộ tensor theo vocab**: Cắt mọi tensor có chiều bằng `262,144` (tối thiểu `embed_tokens.weight`, `embed_tokens_per_layer.weight`, và `lm_head.weight` nếu không tied) dọc trục `0` bằng chỉ số `kept_old_ids`. Giữ nguyên kiểu `bfloat16` và giá trị trọng số gốc, tuyệt đối không khởi tạo lại ngẫu nhiên.
- [ ] **2. Bảo toàn Weight Tying**: Kiểm tra `tie_word_embeddings` trong `config.json`; nếu `true`, đảm bảo khi lưu `safetensors` không vô tình nhân đôi `lm_head.weight` hoặc làm gãy tham chiếu shared tensor.
- [ ] **3. Remap đồng bộ `config.json` ở cả tầng Root lẫn `text_config` (Bắt buộc)**:
  - Cập nhật `vocab_size = new_vocab_size` và `vocab_size_per_layer_input = new_vocab_size` (nếu trường này tồn tại).
  - Remap toàn bộ special token IDs (`pad_token_id`, `bos_token_id`, `eos_token_id`, `unk_token_id`, `image_token_id`, `audio_token_id`, `video_token_id`, `boi_token_id`, `eoi_token_id`, `boa_token_id`, `eoa_token_id`, plus quét đệ quy mọi khóa có đuôi `_token_id` / `_token_index` hoặc mảng `eos_token_id` dạng list) ở **cả tầng root của `config.json` lẫn bên trong object lồng `text_config` (nếu có)**.
  - Cập nhật đồng bộ các token ID tương ứng trong `generation_config.json`.
  - Xác nhận bằng assertion không còn bất kỳ ID nào `>= new_vocab_size` trong mọi file config JSON.
- [ ] **4. Cập nhật `tokenizer.json` và `tokenizer_config.json`**:
  - Trong `tokenizer.json`: thay `model.vocab` bằng bảng ánh xạ `{token_str: new_id}` mới (đảm bảo kích thước bằng đúng `new_vocab_size` và liên tục `0..new_vocab_size-1`), thay `model.merges` bằng danh sách merge đã đóng và giữ nguyên thứ tự, đồng thời remap trường `"id"` của mọi phần tử trong `added_tokens`.
  - Trong `tokenizer_config.json`: remap toàn bộ key chuỗi trong `added_tokens_decoder` sang `str(new_id)` và đối chiếu nhất quán các trường special token.
- [ ] **5. Xử lý dứt điểm `tokenizer.model` (Bắt buộc)**:
  - Kiểm tra xem repo nguồn có chứa `tokenizer.model` (SentencePiece protobuf) hay không.
  - **Phương án ưu tiên**: Xóa bỏ `tokenizer.model` khỏi checkpoint đầu ra và xác minh cả `AutoTokenizer`/`AutoProcessor` (`transformers`) lẫn `convert_hf_to_gguf.py` (`llama.cpp`) đọc duy nhất bảng từ vựng và merges từ `tokenizer.json` đã phẫu thuật.
  - **Phương án dự phòng** (chỉ khi phát hiện consumer bắt buộc yêu cầu `.model`): Rebuild lại protobuf `tokenizer.model` khớp chính xác từng `new_id` và `token_str` từ `tokenizer.json` mới.
  - Tuyệt đối không để tồn tại `tokenizer.model` 262,144 token cũ song song với `tokenizer.json` đã cắt.
- [ ] **6. Cập nhật `model.safetensors.index.json`**: Cập nhật `weight_map` và `metadata.total_size` nếu trọng số được chia shard.

---

## 5. 4 Cổng Kiểm chứng Bắt buộc (Fail-Fast Verification Gates)

Không chuyển sang Bước 1–2 nếu bất kỳ cổng nào dưới đây thất bại:

1. **Cổng 1 — Tokenizer Round-Trip & Policy Verification**:
   - Kiểm tra `decode(encode(x)) == x` không mất mát trên bộ test chuẩn gồm: tiếng Anh kỹ thuật, tiếng Việt đầy đủ thanh điệu & dấu `đ/Đ`, code nhiều ngôn ngữ (Python, C++, JS/TS, HTML/CSS, SQL), thụt lề nhiều space/tab/newline, và ASCII emoticons (`:)`, `:)))`, `._.`, `.-.`, `=))`).
   - Kiểm tra câu chat đầy đủ template với `<|turn>`, `<turn|>`, `<|think|>`, `<|channel>`, `<channel|>`, `<|tool_call>`, `<|image|>`, `<|audio|>`, `[multimodal]`: mỗi special token phải mã hóa thành **đúng 1 ID duy nhất**, khớp 100% với ID mới trong cả tầng root lẫn `text_config` của `config.json`.
   - Kiểm tra chính sách cấm Unicode Emoji: xác nhận các emoji thử nghiệm (`🔥`, `😂`, `🚀`, chuỗi ZWJ) không có trong `vocab`, khi encode sẽ rơi về chuỗi byte-fallback `<0x00>..<0xFF>`, không sinh `<unk>` và không gây exception.

2. **Cổng 2 — Logit Equivalence so với Mô hình Gốc**:
   - Nạp lần lượt (hoặc đối chiếu theo batch) mô hình gốc 262,144 và mô hình đã cắt với cùng danh sách prompt EN / VI / Code / Thinking.
   - Lấy phân phối logits bước kế tiếp của mô hình gốc tại đúng các chỉ số `kept_old_ids`.
   - Ngưỡng đạt: `max_abs_diff < 1e-3` (ở `bfloat16`) và `argmax` dự đoán token kế tiếp trùng khớp 100% trên toàn bộ bộ prompt kiểm tra.

3. **Cổng 3 — End-to-End `Gemma4ForConditionalGeneration` Load & Generate**:
   - Nạp sạch checkpoint đã phẫu thuật từ thư mục output bằng `AutoProcessor.from_pretrained(...)` và `Gemma4ForConditionalGeneration.from_pretrained(...)`.
   - Chạy `generate()` trên prompt song ngữ Anh–Việt hỏi về code có kích hoạt kênh native thinking (`<|think|>`), xác minh không lỗi shape ở `embed_tokens`, `embed_tokens_per_layer` hay `lm_head`.
   - Chạy kiểm tra forward/shape của Vision & Audio encoder (nếu có mẫu ảnh/audio nhỏ) để chứng minh cầu nối `[multimodal]`, `<|image|>`, `<|audio|>` vẫn nguyên vẹn.

4. **Cổng 4 — Dry-Run `llama.cpp` GGUF + `--mmproj` Export**:
   - Dùng bản `llama.cpp` mới (hỗ trợ kiến trúc Gemma 4 và unified/vision-audio `mmproj`, hậu tháng 6/2026) chạy thử:
     - `python convert_hf_to_gguf.py <pruned_dir> --outfile model-text-code-step0.gguf --outtype bf16` rồi quant thử `Q4_K_M`.
     - `python convert_hf_to_gguf.py <pruned_dir> --mmproj --outfile mmproj-vision-audio-step0.gguf --outtype bf16`.
   - Chạy smoke test suy luận text/code với `llama-cli` hoặc `llama-server` (và gắn `--mmproj` khi kiểm tra ảnh/audio). Ghi nhận dung lượng GGUF và mức VRAM tiêu thụ so với Baseline Brain `brain/Gemma-4-E2B-IT-AB.gguf`.
   - Lưu ý dự phòng: Nếu `convert_hf_to_gguf.py --mmproj` gặp lỗi ở phần Audio của Gemma 4 tại thời điểm chạy, ghi nhận rõ lỗi, kiểm chứng xuất riêng Vision `mmproj` trước và giữ nguyên Audio encoder trong `safetensors` theo ADR-0001.

---

## 6. Đẩy Artifact & Ràng buộc chuyển giao sang Bước 1–2

1. **Lưu trữ lên Hugging Face Private Repo**:
   - Đẩy thư mục checkpoint đã qua đủ 4 cổng kèm `id_map.json` và `pruning_report.json` lên Hugging Face Private Repo.
   - Tuyệt đối không commit checkpoint, GGUF hoặc dữ liệu corpus vào GitHub.
2. **Ràng buộc chuyển giao sang Bước 1–2 (Synthetic Distillation)**:
   - Dữ liệu sinh từ Gemini API phải dùng đúng cấu trúc native thinking tokens của Gemma 4 (`<|think|>`, `<|channel>`, `<channel|>`), tuyệt đối không dùng thẻ `<thinking>...</thinking>`.
   - Áp luật kiểm duyệt cứng ở bước hậu xử lý dataset: cấm 100% Unicode emoji trên toàn bộ mẫu; chỉ cho phép ASCII emoticons (`:)`, `:)))`, `._.`, `.-.`, `=))`) ở lời thoại chat tự nhiên, cấm tuyệt đối trong kênh suy luận, code block và tool-call JSON.
