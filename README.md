# Brain Surgery

Dự án phẫu thuật từ điển, chưng cất dữ liệu song ngữ Anh–Việt chuyên Code và tinh chỉnh model Gemma 4 E2B phục vụ trợ lý offline `nuclear-code` chạy trên GPU GTX 1650 Ti (4GB VRAM), sử dụng Google Colab T4 (15GB VRAM) để xử lý và huấn luyện.

## Quy trình 4 giai đoạn

0. **Bước 0 — Phẫu thuật từ điển & PLE (`Vocab Surgery`)**: Từ Source Checkpoint `treadon/gemma4-E2B-it-abliterated` (dự phòng: `Austriani/Gemma-4-E2B-it-ARA-Abliterated`), cắt đồng bộ `tokenizer.json`, `embed_tokens`, `lm_head` (nếu tách rời) và `embed_tokens_per_layer` (PLE) từ `262,144` xuống mục tiêu `49,152–65,536` token (EN + VI + Code + ASCII emoticon). Giữ nguyên toàn bộ special tokens và cầu nối multimodal, loại bỏ hoàn toàn `<unused...>` và Unicode emoji.
1. **Bước 1–2 — Chưng cất dữ liệu tổng hợp (Synthetic Distillation)**: Tạo bộ dữ liệu chất lượng cao chuyên về Code và giải thích song ngữ Anh–Việt (giữ nguyên thuật ngữ chuyên ngành tiếng Anh) qua Gemini API, sử dụng native thinking tokens của Gemma 4 và tuân thủ chính sách cấm Unicode emoji.
2. **Bước 3 — Fine-tune QLoRA trên Colab T4**: Huấn luyện bằng Unsloth trên Colab T4 (15GB VRAM), đóng băng hoàn toàn Vision và Audio encoder, chỉ train phần Text/Code cùng các bảng embedding đã cắt.
3. **Bước 4 — Xuất GGUF theo kiến trúc Modular Add-on**: Xuất tách biệt `model-text-code.gguf` (Core Brain mặc định siêu nhẹ cho tác vụ text/code) và `mmproj-vision-audio.gguf` (Multimodal Add-on chỉ mount qua `--mmproj` khi cần xử lý ảnh/âm thanh).

## Phân tách lưu trữ & đồng bộ

- **GitHub**: Chứa mã nguồn, tài liệu kiến trúc, kế hoạch thực thi và Colab notebook (`.gitignore` chặn toàn bộ file weights, GGUF và dataset lớn).
- **Hugging Face Private Repo**: Chứa Source/Pruned `safetensors` checkpoint, synthetic dataset và các file GGUF/mmproj xuất ra để đồng bộ nhanh giữa Colab và máy local.

## Cấu trúc thư mục (Bước 0)

```text
brain_surgery/
├── .gitignore
├── README.md
├── GLOSSARY.md
├── docs/
│   ├── adr/
│   │   ├── 0001-modular-gguf-and-mmproj-addon.md
│   │   ├── 0002-lockstep-ple-vocab-pruning-with-bpe-closure.md
│   │   └── 0003-native-thinking-and-ascii-emoticon-policy.md
│   └── plans/
│       └── step-0-vocab-pruning.md
├── notebooks/               # Colab notebooks (bắt đầu từ Bước 0)
├── src/
│   └── surgery/             # Mã nguồn phẫu thuật từ điển & kiểm chứng
└── brain/                   # Chứa Baseline Brain local và GGUF thành phẩm
```

## Tài liệu liên quan

- Thuật ngữ dự án: [GLOSSARY.md](GLOSSARY.md)
- Quyết định kiến trúc: [ADR-0001](docs/adr/0001-modular-gguf-and-mmproj-addon.md), [ADR-0002](docs/adr/0002-lockstep-ple-vocab-pruning-with-bpe-closure.md), [ADR-0003](docs/adr/0003-native-thinking-and-ascii-emoticon-policy.md)
- Kế hoạch chi tiết Bước 0: [step-0-vocab-pruning.md](docs/plans/step-0-vocab-pruning.md)
- Quy ước thư mục model local: [brain/README.md](brain/README.md)
