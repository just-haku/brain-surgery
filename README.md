# Brain Surgery

Dự án phẫu thuật từ điển, chưng cất dữ liệu song ngữ Anh–Việt chuyên Code và tinh chỉnh model Gemma 4 E2B phục vụ trợ lý offline `nuclear-code` (`nc-agent`) chạy trên GPU GTX 1650 Ti (4GB VRAM), sử dụng Google Colab (T4 15GB VRAM cho Bước 0; Colab Pro L4 24GB / A100 40GB cho Bước 2).

## Quy trình các giai đoạn

0. **Bước 0 — Phẫu thuật từ điển & PLE (`Vocab Surgery`) — Đã hoàn tất**: Từ `treadon/gemma4-E2B-it-abliterated`, cắt đồng bộ `tokenizer.json`, `embed_tokens` và `embed_tokens_per_layer` (PLE) từ `262,144` xuống `65,536` token (`haku-nekooo/gemma4-e2b-nuclear-pruned`), đạt `max_abs_diff = 0.0`, `Q4_K_M = 1.594 GB`, `mmproj = 0.919 GB`.
1. **Bước 1 — Chưng cất dữ liệu Agentic & Song ngữ (Synthetic Distillation)**: Sinh quỹ đạo nhiều lượt chuẩn format `<|channel>thought\n...<channel|>` và native tool calling (`read_symbol`, `search_text`, `edit_file`) qua Gemini API; trị 3 lỗi thực chiến của `nc-agent` và kiểm duyệt cứng cấm 100% Unicode emoji.
2. **Bước 2 — Train QLoRA SFT -> ORPO trên Colab Pro (L4/A100)**: Đóng băng hoàn toàn Vision và Audio encoder để bảo toàn `mmproj`; chạy QLoRA SFT (train thêm `embed_tokens`, `lm_head` và `PLE` với learning rate thấp), sau đó bồi 1 vòng ORPO ngắn trên các cặp lỗi thực chiến; xuất `model-text-code.gguf` (`Q4_K_M` / `Q5_K_M`) và kiểm chứng bộ regression suite `nc-agent`.

## Phân tách lưu trữ & đồng bộ

- **GitHub**: Chứa mã nguồn, tài liệu kiến trúc, kế hoạch thực thi và Colab notebook (`.gitignore` chặn toàn bộ file weights, GGUF và dataset lớn).
- **Hugging Face Private Repo**: Chứa Source/Pruned `safetensors` checkpoint (`haku-nekooo/gemma4-e2b-nuclear-pruned`), synthetic dataset và các file GGUF/mmproj xuất ra để đồng bộ nhanh giữa Colab và máy local.

## Cấu trúc thư mục

```text
brain_surgery/
├── .gitignore
├── README.md
├── GLOSSARY.md
├── docs/
│   ├── adr/
│   │   ├── 0001-modular-gguf-and-mmproj-addon.md
│   │   ├── 0002-lockstep-ple-vocab-pruning-with-bpe-closure.md
│   │   ├── 0003-native-thinking-and-ascii-emoticon-policy.md
│   │   └── 0004-two-stage-qlora-orpo-and-native-tool-trajectories.md
│   └── plans/
│       ├── step-0-vocab-pruning.md
│       └── step-1-2-distillation-and-training.md
├── notebooks/               # Colab notebooks (step0_vocab_pruning.ipynb, ...)
├── src/
│   ├── surgery/             # Mã nguồn phẫu thuật từ điển & kiểm chứng Bước 0
│   ├── distill/             # Sinh dữ liệu Gemini API, validator & Gemma 4 serializer
│   └── train/               # Pipeline QLoRA SFT / ORPO & Agentic eval harness
├── tests/                   # Bộ kiểm thử đơn vị cục bộ
└── brain/                   # Chứa Baseline Brain local và GGUF thành phẩm
```

## Tài liệu liên quan

- Thuật ngữ dự án: [GLOSSARY.md](GLOSSARY.md)
- Quyết định kiến trúc: [ADR-0001](docs/adr/0001-modular-gguf-and-mmproj-addon.md), [ADR-0002](docs/adr/0002-lockstep-ple-vocab-pruning-with-bpe-closure.md), [ADR-0003](docs/adr/0003-native-thinking-and-ascii-emoticon-policy.md), [ADR-0004](docs/adr/0004-two-stage-qlora-orpo-and-native-tool-trajectories.md)
- Kế hoạch chi tiết: [Bước 0](docs/plans/step-0-vocab-pruning.md), [Bước 1–2](docs/plans/step-1-2-distillation-and-training.md)
- Quy ước thư mục model local: [brain/README.md](brain/README.md)
