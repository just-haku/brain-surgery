# Kế hoạch Bước 1–2: Synthetic Distillation & QLoRA/ORPO cho Gemma 4 E2B (`nc-agent`)

## 1. Mục tiêu, Đầu vào & Đầu ra

- **Đầu vào (Artifact từ Bước 0)**: Checkpoint `haku-nekooo/gemma4-e2b-nuclear-pruned` trên Hugging Face Private Repo (`65,536` tokens, BF16 `6.08 GB`, `max_abs_diff = 0.0`, `Q4_K_M = 1.594 GB`, `mmproj = 0.919 GB`).
- **Quy ước Baseline & Chuẩn Template**: Mọi đối chiếu `chat_template.jinja`, token suy luận (`<|channel>thought\n...<channel|>`) và cấu trúc gọi tool phải lấy chuẩn trực tiếp từ checkpoint Hugging Face (`haku-nekooo/gemma4-e2b-nuclear-pruned` / `google/gemma-4-E2B-it`). Tệp GGUF cũ trước đây đã deprecated; benchmark local dùng cặp `brain/gemma-4-E2B-it-Q4_K_M.gguf` + `brain/mmproj-F16.gguf` mới pull từ Hugging Face.
- **Mục tiêu thực chiến**: Huấn luyện Core Brain đạt độ tin cậy cao cho `nuclear-code` (`nc-agent`) trên GTX 1650 Ti (4GB VRAM), trị dứt điểm bệnh lặp chữ và **3 lỗi Agentic thực chiến**:
  1. Lẫn ngữ cảnh giữa các lượt hội thoại hoặc đọc lặp lại dữ liệu đã có trong `## SCRATCHPAD` (context bleed).
  2. Ảo giác tham số tool (như truyền `start_line`/`content` vào `edit_file` thay vì `target`/`replacement`), lặp vòng JSON lồng nhau trong `"final"`, hoặc rò JSON thô ra phần chat người dùng.
  3. Tự ý đổi kiểu dữ liệu khi sửa code (ví dụ tự ý biến `i32` thành `f64` trong Rust hoặc phá signature).
- **Môi trường chạy**: Sinh & kiểm duyệt dữ liệu bằng **Gemini Batch API** (kết hợp `gemini-2.5-pro` và `gemini-2.5-flash`); huấn luyện và xuất GGUF trên **Google Colab Pro (GPU L4 24GB hoặc A100 40GB)** có hỗ trợ BF16 gốc.
- **Đầu ra (đẩy lên HF Private Repo & kéo GGUF về `brain/`)**:
  - `datasets/sft_trajectories.jsonl` (~3,500–5,000 mẫu sạch) và `datasets/orpo_pairs.jsonl` (~500–800 cặp).
  - Checkpoint BF16 đã merge sau Stage 2A (SFT) -> Stage 2B (ORPO).
  - File thành phẩm: `model-text-code-v1-Q4_K_M.gguf` (~1.6 GB), `model-text-code-v1-Q5_K_M.gguf` (~1.9 GB) và `mmproj-vision-audio-v1-bf16.gguf` (~0.92 GB).

---

## 2. Chuẩn Wire Format Gemma 4 & Hợp đồng `nc-agent` (Đã xác minh từ Source Code)

### 2.1. Hợp đồng Thực tế của `nc-agent` (`D:\Coding\nuclear_code`)

Qua đọc trực tiếp `crates/nc-agent/src/context.rs`, `orchestrator.rs`, `providers.rs` và `prompts/caveman_system.md`:

- **Giao thức gọi `llama-server --jinja`**: `nc-agent` gửi `POST /v1/chat/completions` kèm `messages: [system, user]` và tham số `grammar` (`tool_call_gbnf()`, `planner_gbnf()`, hoặc `final_only_gbnf()`). `llama-server --jinja` áp dụng `chat_template.jinja`, tách phần `<|channel>thought\n...<channel|>` vào `reasoning_content` (khi bật `enable_local_thinking`) và tách phần JSON hợp lệ theo GBNF vào `choices[0].message.content`.
- **Cấu trúc Prompt `nc-agent` (`assemble_context`)**:
  - `system`: Nội dung `caveman_system.md` + `Role: CODER | PLANNER | REVIEWER`.
  - `user`: Các block ngăn nắp theo thứ tự: `## REPO_MAP`, `## LEDGER`, `## TASK`, `## SCRATCHPAD` (gồm `[step N tool]\n...` hoặc receipt bị thu gọn `[step N tool -> receipt]`), và `## NEXT`.
- **Định dạng JSON Output Bắt buộc**:
  - Bước gọi tool (`CODER`): `{"thought":"...","action":{"tool":"<name>","args":{...}}}`
  - Bước trả lời cuối (`CODER` / `REVIEWER`): `{"thought":"...","final":"plain concise answer"}` (tuyệt đối không lồng thêm JSON bên trong `"final"`).
  - Bước lập kế hoạch (`PLANNER`): `{"goal":"...","decisions":[...],"tasks":[{"id":"T1","title":"...","target_files":["..."],"verify_cmd":null}]}` (tối đa 4 micro-tasks).
- **Schema Chính xác của 8 Tool trong `orchestrator.rs` (Trọng tâm 3 tool chính + 5 tool phụ)**:
  1. `read_symbol`: `{"path": str, "symbol": str}` — cả 2 bắt buộc.
  2. `search_text`: `{"query": str, "ext"?: str, "max_hits"?: int}` — `query` bắt buộc; `ext` lọc đuôi file (ví dụ `"rs"`, `"py"`), `max_hits` mặc định `15`.
  3. `edit_file`: `{"path": str, "target": str, "replacement": str}` — cả 3 bắt buộc, `target` phải là chuỗi text khớp chính xác văn bản đang tồn tại (cấm tuyệt đối bịa khóa `start_line`, `end_line`, `content`, `new_content`, `old_str`, `new_str`).
  4. Tool phụ đi kèm cùng `tool_call_gbnf()`: `read_outline(path: str)`, `read_slice(path: str, start_line: int, end_line: int)`, `list_files(dir?: str, max_entries?: int)`, `write_file(path: str, content: str)`, `run_command(cmd: str)`.

### 2.2. Chuẩn Wire Format Gemma 4 & Bẫy `strip_thinking()`

Dựa trên `chat_template.jinja` của checkpoint chuẩn trên Hugging Face (`haku-nekooo/gemma4-e2b-nuclear-pruned` / `google/gemma-4-E2B-it`):

- **System turn**: `<bos><|turn>system\n<|think|>{system_prompt}<turn|>\n` (bỏ `<|think|>` khi `enable_thinking=False` — tập train bao gồm ~70% bật thinking và ~30% tắt thinking khớp cờ `enable_local_thinking` của `nc-agent`).
- **User turn**: `<|turn>user\n{user_content}<turn|>\n`.
- **Model turn**: `<|turn>model\n<|channel>thought\n{suy_luận_ngắn_gọn}<channel|>{model_content}<turn|>\n` (khi `enable_thinking=False` thì bỏ khối `<|channel>thought...<channel|>`).
- **Cảnh báo Bẫy `strip_thinking()`**: Macro `strip_thinking(text)` trong `chat_template.jinja` tự động **xóa toàn bộ `<|channel>thought...<channel|>`** khỏi tin nhắn `model` khi gọi `tokenizer.apply_chat_template()`. Vì vậy Step 1–2 bắt buộc dùng `render_gemma4_training_sample()` trong [`formatter.py`](../../src/distill/formatter.py) kèm **assistant-only loss masking** (`labels = -100` trên mọi turn `system` và `user`).

---

## 3. Bước 1 — Synthetic Distillation Blueprint (Gemini Batch API)

### 3.1. Phân luồng Model & Cơ cấu SFT (~3,500–5,000 mẫu)

1. **Nhóm A (35% — ~1,400–1,750 mẫu) [`gemini-2.5-pro`]: Quỹ đạo `nc-agent` (`Agentic Trajectory`)**
   - Mô phỏng prompt `## REPO_MAP / ## LEDGER / ## TASK / ## SCRATCHPAD / ## NEXT` với JSON output khớp 100% `tool_call_gbnf()` và `planner_gbnf()`.
   - Đánh trực diện vào 3 lỗi:
     - *Cô lập ngữ cảnh & tôn trọng SCRATCHPAD*: Khi chuyển file/symbol ở `## TASK` mới hoặc đã có `[step N tool -> receipt]` trong `## SCRATCHPAD`, không lặp lại read thừa và không lẫn symbol cũ.
     - *Kỷ luật tham số & chống lồng JSON*: `edit_file` luôn dùng đúng `path`, `target`, `replacement`; `"final"` luôn là văn bản sạch, không nhét JSON lồng nhau.
     - *Bảo toàn kiểu dữ liệu & signature*: Giữ nguyên `i32` trong Rust, `int32_t` trong C++, interface TS khi sửa logic bằng `edit_file` hoặc `write_file`.

2. **Nhóm B (25% — ~900–1,250 mẫu) [`gemini-2.5-flash`]: Giải thích Code & Debug Song ngữ Anh–Việt**
   - Giải thích lỗi biên dịch (`borrow checker`, `lifetime`, `segfault`, `deadlock`, `race condition`, tối ưu SQL, giải thuật) bằng **tiếng Việt mạch lạc, giữ nguyên thuật ngữ chuyên ngành tiếng Anh**.
   - `<|channel>thought\n...<channel|>` cô đọng (tối đa ~150–250 token), gạch ý dứt khoát để trị bệnh lặp chữ.

3. **Nhóm C (15% — ~550–750 mẫu) [`gemini-2.5-pro`]: Sinh & Refactor Code Trực tiếp (Single-turn)**
   - Sửa hoặc viết code tối giản (`ponytail`), giữ nguyên kiểu dữ liệu và kèm test biên ngay khi người dùng đã dán đủ code.

4. **Nhóm D (15% — ~550–750 mẫu) [`gemini-2.5-flash`]: Chống Ảo giác & Hỏi lại khi Thiếu thông tin**
   - Khi thiếu file hoặc ký hiệu chưa có trong `## SCRATCHPAD`/`## REPO_MAP`: ở chế độ `nc-agent` gọi ngay `search_text`/`read_symbol`; ở chế độ chat hỏi lại 1 câu ngắn gọn, không bịa API hay đường dẫn.

5. **Nhóm E (10% — ~350–500 mẫu) [`gemini-2.5-flash`]: Chat Dev Tự nhiên với ASCII Emoticon có kiểm soát**
   - Giao tiếp kỹ sư phần mềm (EN/VI), cho phép dùng ASCII emoticon (`:)`, `:)))`, `._.`, `.-.`, `=))`) **chỉ trong văn bản trả lời giao tiếp tự nhiên**, không lạm dụng.

### 3.2. Bộ Kiểm duyệt Cứng Tự động (`src/distill/validator.py`)

Mọi mẫu sinh từ Gemini đều phải vượt qua bộ kiểm duyệt quyết định trước khi vào tập train:

- **Cấm 100% Unicode Emoji**: Dùng lại `is_emoji_or_banned_symbol()` từ [`vocab_filter.py`](../../src/surgery/vocab_filter.py) để loại bỏ mọi mẫu chứa emoji, pictograph, ZWJ (`U+200D`) hoặc variation selector (`U+FE00..U+FE0F`) ở mọi trường.
- **Vùng cấm ASCII Emoticon**: Chặn tuyệt đối ASCII emoticon trong `<|channel>thought...<channel|>`, mọi code block ```` ```...``` ````, và toàn bộ đầu ra chế độ `nc_agent`. Chỉ cho phép ở văn bản prose của chế độ `chat`.
- **Kiểm tra GBNF Envelope & Schema `nc-agent`**: Parse JSON nghiêm ngặt, kiểm tra `thought` + (`action` XOR `final`) hoặc schema `PLANNER`, kiểm tra đúng tên tool + kiểu tham số của 8 tool trong `orchestrator.rs`, chặn khóa bịa (`start_line`/`content` trong `edit_file`) và chặn lồng JSON trong `"final"`.
- **Chống lặp & Độ dài**: Chặn vòng lặp n-gram trong thought/answer; check cân bằng thẻ Gemma 4; dùng tokenizer `haku-nekooo/gemma4-e2b-nuclear-pruned` kiểm tra độ dài `<= 4096` token và `0` `<unk>`; loại mẫu trùng bằng hash.

### 3.3. Tập `Preference Pair` cho ORPO (~500–800 cặp) [`gemini-2.5-pro` + lỗi thật Bước 0]

- Ghép cặp lỗi thực tế (từ model Bước 0 trên test `nc-agent` + nhiễu chủ đích từ `gemini-2.5-pro`) làm `rejected` và bản chuẩn sạch làm `chosen` cho 4 nhóm lỗi:
  1. `context_bleed`: lẫn file/symbol giữa lượt cũ và mới, hoặc đọc lại thứ đã có trong `## SCRATCHPAD`.
  2. `tool_schema_or_json_leak`: bịa `start_line`/`content` cho `edit_file`, lồng JSON vào `"final"`, hoặc bọc code fence quanh JSON.
  3. `type_mutation`: tự ý đổi `i32 -> f64` hoặc phá chữ ký hàm khi sửa code.
  4. `repetition_or_emoji`: lặp chữ trong thought/answer hoặc lọt Unicode emoji.

---

## 4. Bước 2 — Chiến lược Huấn luyện trên Colab Pro (L4 24GB / A100 40GB)

### 4.1. Bất biến Đóng băng Đa phương thức (Multimodal Freeze Invariant)

- Khóa cứng `requires_grad = False` cho toàn bộ tham số `vision_tower`, `audio_tower`, và `multi_modal_projector`.
- Băm SHA-256 toàn bộ tensor Vision/Audio trước và sau khi merge adapter để bảo đảm tệp `mmproj-vision-audio.gguf` (`0.919 GB`) hoàn toàn đồng bộ với Core Brain mới.

### 4.2. Stage 2A — QLoRA SFT (`Unsloth` / `TRL SFTTrainer`, BF16 gốc)

- **Nạp model**: `haku-nekooo/gemma4-e2b-nuclear-pruned` (4-bit NF4 compute `torch.bfloat16` trên L4 24GB, hoặc BF16 LoRA trên A100 40GB).
- **Cấu hình LoRA & Embedding/PLE**:
  - `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`, `r = 32`, `lora_alpha = 64`, `lora_dropout = 0`, `lr = 1.5e-4`, `cosine` scheduler, `warmup_ratio = 0.05`.
  - `modules_to_save`: `["embed_tokens", "embed_tokens_per_layer", "lm_head"]` với `embedding_lr = 1.5e-5` (nhỏ hơn `10×` so với LoRA, kiểm tra giữ `tie_word_embeddings` sạch khi lưu).
- **Tham số chuỗi & Loss**: `max_seq_length = 4096`, `assistant-only loss masking` (chỉ tính loss trên lượt `model` sau `<|turn>model\n`), `adamw_8bit`, `effective_batch_size = 16`, `2–3 epochs` kèm eval set ~200 mẫu để early stopping.
- Merge adapter Stage 2A về BF16 `safetensors` và kiểm tra hash Vision/Audio.

### 4.3. Stage 2B — Một vòng ORPO Ngắn (`TRL ORPOTrainer`)

- Nạp checkpoint BF16 đã merge từ Stage 2A; gắn LoRA Attention/MLP (`r = 16`, `lora_alpha = 32`); **đóng băng hoàn toàn `embed_tokens`, `lm_head` và `embed_tokens_per_layer` (PLE)**.
- Train `1 epoch` (`beta = 0.1`, `lr = 8e-6`, `max_length = 4096`, `max_prompt_length = 2560`) trên `500–800` cặp lỗi thực chiến.
- Merge adapter thành checkpoint BF16 cuối cùng, kiểm tra lại SHA-256 Vision/Audio và đẩy lên HF Private Repo.

---

## 5. Bước 2C — Xuất GGUF & Cổng Kiểm chứng Agentic

1. **Xuất GGUF tách modular (ADR-0001)**:
   - `model-text-code-v1-Q4_K_M.gguf` (~1.6 GB, mặc định cho GTX 1650 Ti 4GB VRAM).
   - `model-text-code-v1-Q5_K_M.gguf` (~1.9 GB, dự phòng chất lượng cao).
   - `mmproj-vision-audio-v1-bf16.gguf` (~0.92 GB).
2. **Bộ Kiểm chứng Hồi quy Agentic (`src/train/eval_agentic.py`, 25–30 test case)**:
   - Chấm điểm so sánh đối chiếu trực tiếp với baseline mới trong `brain/` (`gemma-4-E2B-it-Q4_K_M.gguf` + `mmproj-F16.gguf`) qua `llama-server --jinja` + `tool_call_gbnf()`:
     1. Cô lập ngữ cảnh đa lượt & không lặp `## SCRATCHPAD`.
     2. Kỷ luật schema 8 tool (`read_symbol`, `search_text`, `edit_file`...), `0` JSON lồng trong `"final"`.
     3. Bảo toàn kiểu dữ liệu (`i32` trong Rust, `int32_t` trong C++, strict TS).
     4. Giải thích song ngữ EN–VI chuẩn thuật ngữ, không lặp chữ, `0` Unicode emoji, tuân thủ vùng cấm ASCII emoticon.
     5. Smoke test mount `--mmproj` với ảnh code mẫu.

---

## 6. Cấu trúc Mã nguồn & Quyết định Đã Chốt

- `src/distill/`: `nc_schema.py` (schema thật từ `nc-agent` + `validate_nc_envelope`), `formatter.py` (`render_gemma4_training_sample` + `render_gemma4_orpo_pair`), `validator.py` (chặn emoji/vùng cấm emoticon/anti-loop/token length), `prompts.py` (sinh prompt pool định tuyến `gemini-2.5-pro` & `gemini-2.5-flash`), `gemini_batch.py` (`google-genai` Batch API JSONL + Sync smoke + parser), `cli.py`.
- `src/train/`: Pipeline Stage 2A SFT + Stage 2B ORPO + SHA-256 Vision/Audio verifier + `eval_agentic.py`.
- `notebooks/`: `step1_distill.ipynb` (nếu cần chạy từ Colab), `step2_train_qlora_orpo.ipynb`.
- **Các điểm đầu vào đã chốt xong**:
  1. Schema tool trích trực tiếp từ `D:\Coding\nuclear_code\crates\nc-agent\src\orchestrator.rs` & `context.rs`.
  2. `nc-agent` gọi `llama-server --jinja` qua `/v1/chat/completions` với GBNF grammar (`providers.rs`).
  3. Dùng `gemini-2.5-pro` (nhóm Agentic/Code/ORPO) + `gemini-2.5-flash` (Giải thích/Hỏi lại/Chat) qua Gemini Batch API (`google-genai`).
