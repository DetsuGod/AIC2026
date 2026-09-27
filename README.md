<div align="center">

# 🎬 AIC 2026: Multimodal Video Retrieval & Interactive Search Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CUDA-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![LanceDB](https://img.shields.io/badge/LanceDB-476k%20Keyframes-8A2BE2)](https://lancedb.github.io/lancedb/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A production-grade multimodal interactive video retrieval system for the **Ho Chi Minh City AI Challenge (AIC 2026)** — Final Round. Supports Known-Item Search (KIS), Visual Question Answering (QA), and Temporal Reasoning & Keyframe Extraction (TRAKE).*

[Tiếng Việt](README_VN.md) • [English](README.md) • [Architecture](#️-system-architecture) • [Quickstart](#-quickstart) • [Deployment Notes](#-deployment-notes-for-teammates)

</div>

---

## 🌟 Highlights

| Feature | Details |
|---------|---------|
| **Visual Encoder** | `Qwen/Qwen3-VL-Embedding-2B` — 2048-dim hyperspherical vectors |
| **Dataset** | **1,487 videos**, **476,140 keyframes** (Batch 1 + Batch 2) |
| **Video Formats** | `.mp4` (Batch 1 groups L) and `.mp4`/`.mov` (Batch 2 groups M/S/N) |
| **Vector DB** | LanceDB with 3 tables: `keyframes_k1r`, `video_shots`, `asr_segments` |
| **ASR** | Faster-Whisper Large-V3 + TF-IDF N-gram index (Vietnamese/English) |
| **Fusion** | RRF (Reciprocal Rank Fusion) or Weighted across Visual/ASR/Shot signals |
| **Contest Client** | DRES Live submission (KIS / QA / TRAKE) with offline dry-run mode |
| **UI Style** | Dark Slate — Linear / Raycast Pro Minimalist |

---

## 🏛️ System Architecture

```
Raw Query (Vietnamese / English)
        │
        ▼
[Gemini AI Parser]
  Extracts: visual_query / speech_query / qa_question / multi-scene events
        │
        ▼
[Stage 1 — Multi-Modal Fast Search (Top K1 candidates)]
  ├─ Visual: Qwen3-VL-Embedding-2B → LanceDB cosine search (476,140 KF)
  ├─ ASR:    Faster-Whisper + TF-IDF N-gram (16,609 speech segments)
  ├─ Shot:   Qwen3-VL Video 2048D → LanceDB video_shots (186,102 shots)
  ├─ Caption: TF-IDF index (51,650 matched captions — optional, default OFF)
  └─ Dedup: Smart intra-video + Cross-video deduplication
        │
        ▼
[Stage 2 — TRAKE / Multi-Scene Temporal Search]
  ├─ Temporal DP:  t(E1) < t(E2) < … < t(En) monotonic constraint
  ├─ Alpha-Fusion: 0.35 × GlobalScore + 0.65 × DPScore
  └─ Diverse Beam Search: K-paths per candidate video
        │
        ▼
[Interactive Web UI — FastAPI + Vanilla JS]
  ├─ KIS / QA / TRAKE submission via DRES live API
  ├─ Ctrl+Click on video player → instant submission
  ├─ Alt+G shortcut → jump to any video/keyframe
  └─ Taxonomy filter (7 categories) + Timeline inspector
```

---

## 📂 Repository Structure

```
2026-AIC/
├── Data/                                  # [NOT IN GIT] Received from team leader
│   ├── aic_lancedb/                       # LanceDB vector DB (~4.4 GB)
│   │   ├── keyframes_k1r.lance/           # 476,140 keyframes × 2048D
│   │   ├── video_shots.lance/             # 186,102 video shots × 2048D
│   │   └── asr_segments.lance/            # 16,609 speech segments
│   ├── batch 1/
│   │   ├── Custom_Keyframes/              # JPG keyframe images (groups L)
│   │   └── video/                         # L*.mp4 source videos
│   ├── batch 2/
│   │   ├── Custom_Keyframes/              # JPG keyframe images (groups M/S/N)
│   │   └── video/                         # M*.mp4 / S*.mp4 / N*.mov source videos
│   ├── asr_mapping.json                   # Whisper ASR segments JSON
│   ├── video_fps_mapping.json             # FPS for all 1,487 videos
│   ├── video_categories.json              # Taxonomy for 1,487 videos
│   ├── kf_shot_map.json                   # Keyframe ↔ Shot bidirectional map
│   └── usearch_qwen_index_mapping_k1r.json# Vector index → video_id/frame_id map
│
├── code/
│   ├── local_retrieval/                   # Core Python search engine
│   │   ├── venv/                          # ⭐ Virtual environment (NOT in Git)
│   │   ├── config.py                      # Centralised config + .env loader
│   │   ├── search_engine.py               # Multimodal hybrid retrieval engine
│   │   ├── text_encoder.py                # Qwen3-VL text + image encoder
│   │   ├── gemini_parser.py               # LLM query decomposition
│   │   ├── asr_engine.py                  # Whisper ASR hybrid search
│   │   ├── dres_client.py                 # DRES live contest client
│   │   ├── caption_engine.py              # Caption TF-IDF retrieval (optional)
│   │   └── qwen_reranker.py               # GPU Qwen-VL reranker (optional)
│   └── web/
│       ├── backend/main.py                # FastAPI server (search, stream, DRES)
│       └── frontend/
│           ├── index.html                 # Web UI
│           ├── app.js                     # Frontend controller
│           └── style.css                  # Dark Slate UI
│
├── verify_setup.py                        # 1-click environment health check
├── run_app.bat                            # 1-click launcher (Windows)
├── run_app.sh                             # 1-click launcher (Linux/macOS)
├── requirements.txt                       # Python dependencies
├── .env.example                           # Environment variables template
├── PROJECT_CONTEXT.md                     # Full architecture & algorithm docs
└── code/DATA_DICTIONARY.md                # Data assets reference guide
```

---

## 🚀 Quickstart

### Step 1 — Clone the repository

```bash
git clone https://github.com/liem-2006/aic2026.git
cd aic2026
```

### Step 2 — Create the virtual environment

The venv **must** live at `code/local_retrieval/venv/` (the launcher scripts look here first):

```bash
# Windows
cd code\local_retrieval
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r ..\..\requirements.txt
cd ..\..

# Linux / macOS
cd code/local_retrieval
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r ../../requirements.txt
cd ../..
```

### Step 3 — Configure environment variables

```bash
cp .env.example .env   # Linux/macOS
copy .env.example .env # Windows
```

Edit `.env`:
```ini
# Google Gemini API key (query parsing — optional but recommended)
GEMINI_API_KEY=AIzaSy...

# DRES contest server (Final Round submission)
DRES_SERVER_URL=https://eventretrieval.one
DRES_USERNAME=team_xxx
DRES_PASSWORD=your_password
```

### Step 4 — Place data assets

Receive from your team leader and place:
- `Data/aic_lancedb/` — LanceDB vector database (~4.4 GB, **required**)
- `Data/batch 1/Custom_Keyframes/` — Batch 1 keyframe images
- `Data/batch 1/video/` — Batch 1 source `.mp4` videos
- `Data/batch 2/Custom_Keyframes/` — Batch 2 keyframe images (M/S/N groups)
- `Data/batch 2/video/` — Batch 2 source videos (`.mp4` for M/S, `.mov` for N)
- `Data/asr_mapping.json`, `Data/video_fps_mapping.json`, `Data/video_categories.json`

### Step 5 — Verify system health

```bash
python verify_setup.py
```

All critical items should show `[ĐẠT - OK]`. Warnings (yellow) are non-blocking.

### Step 6 — Launch

```bash
# Windows (1-click, auto-opens browser)
run_app.bat

# Linux / macOS
bash run_app.sh

# Manual
code/local_retrieval/venv/Scripts/python code/web/backend/main.py
```

Open **`http://127.0.0.1:8000`** in your browser.

> ⚠️ Wait for the `READY` badge to appear in the sidebar before searching. Backend warms up LanceDB + ASR index in the background (~15–60 s depending on hardware).

---

## 🖥️ Key Features

### Search Modes
| Mode | Trigger | Description |
|------|---------|-------------|
| **KIS Search** | `🔍 SEARCH NOW` | Single-scene known-item search |
| **Multi-Scene** | `🎬 MULTI-SCENE` | Chronologically ordered multi-scene search |
| **TRAKE** | `⏱️ SEARCH TRAKE` | Temporal event sequence search (TARS Monotonic DP) |

### Submission Modes (DRES Live)
| Mode | Ctrl+Click behaviour |
|------|---------------------|
| **KIS** | Submit instantly with auto-converted ms timestamp |
| **QA** | Open QA answer form, then submit |
| **TRAKE** | Select event sequence (E1 → E2 → …) then submit |

### Other Features
- **Alt+G** — Jump to any video/keyframe outside Top-K results
- **Timeline Inspector** — Browse all keyframes of a video chronologically
- **Lightbox** — Pan & Zoom inspection (up to 1000%)
- **Taxonomy Filter** — 7 categories: `am_thuc`, `tin_tuc`, `day_hoc`, `du_lich_van_hoa`, `mua_lan`, `dua_xe_dap`, `camera_giao_thong`
- **RRF Fusion** or **Weighted Fusion** — switchable in sidebar
- **Smart Dedup** — intra-video + cross-video intro/ad deduplication
- **DRES Offline Dry-Run** — test the full workflow without a live contest connection

---

## 📦 Deployment Notes for Teammates

> Read `PROJECT_CONTEXT.md` and `code/DATA_DICTIONARY.md` for full algorithm details.

**Minimum required to search (no streaming):**
- LanceDB `Data/aic_lancedb/` with `keyframes_k1r` table
- `Data/video_fps_mapping.json`
- `Data/asr_mapping.json`

**Required for image preview:**
- `Data/batch 1/Custom_Keyframes/` (groups L)
- `Data/batch 2/Custom_Keyframes/` (groups M/S/N)

**Required for video streaming (Timeline, DRES ms):**
- `Data/batch 1/video/*.mp4`
- `Data/batch 2/video/*.mp4` and `*.mov`

**DRES submission:**
- Fill `DRES_USERNAME`, `DRES_PASSWORD`, `DRES_SERVER_URL` in `.env`
- Enter the contest **Run ID** in the DRES Live Bar on the web UI

---

## ❓ Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `DLL load failed while importing _lancedb` | Missing C++ Runtime on Windows | Install [VC++ 2022 Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| `WinError 10048` (port in use) | Another backend already running on port 8000 | Kill the previous process or use a different port |
| Keyframe images show 404 | Wrong keyframe directory path | Check `Data/batch 1/Custom_Keyframes/{VIDEO_ID}/{FRAME}.jpg` structure |
| TRAKE search very slow | `Top Videos` value too high | Reduce `Top Videos` in TRAKE params panel (default 10) |
| Gemini parser returns empty | No API key configured | Add `GEMINI_API_KEY` to `.env`; fallback parser still works without it |
| No GPU, slow encoding | CPU-only machine | System falls back to CPU automatically — expect 5–20× slower search |

---

## 👥 Team & References

Developed for **AIC 2026 (Ho Chi Minh City AI Challenge)** — Final Round.

- *Qwen3-VL-Embedding* (arXiv:2601.04720)
- *TARS Monotonic DP for Temporal Reasoning* (AIC 2025, 93.15% accuracy)
- *LanceDB Embedded Vector Architecture*
- *Faster-Whisper Large-V3 ASR*

---

## 📄 License

Distributed under the [MIT License](LICENSE). Competition video data and proprietary vector indices are managed under private licenses and are **not included** in this repository.
