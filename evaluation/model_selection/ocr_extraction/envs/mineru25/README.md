# mineru25 env, DONE

Isolated venv for **MinerU2.5** (`opendatalab/MinerU2.5-2509-1.2B`, 1.2B params, Paddle family , 
grouping follows the run plan's family label; the model itself is Qwen2-VL-architecture via
`transformers`, not PaddlePaddle), per docs/MODEL_SELECTION.md Section 2. Built with `uv`. README
gives no pinned versions, installed latest via `mineru-vl-utils[transformers]`, then pinned
`torch==2.7.1+cu118`/`torchvision==0.22.1+cu118` after the same CPU-only-wheel gotcha every
candidate so far has hit. See `requirements.txt`.

**Run order (smallest-to-largest, run plan Section 4): 3 of 7, COMPLETE.**

Real numbers:
- Load time: 10.5s, peak VRAM 2.15 GB, fits 8GB gate: **YES**
- 18/18 gold pages OK via `MinerUClient.two_step_extract()`, structured `ExtractResult` output
  (bbox + type + content per region), spot-checked accurate on multiple pages, including
  correctly identifying an answer-key grid page as `table`/`title`/`table_caption` types.

Raw output: `../raw/mineru25/*.json` (gitignored, same policy as prior candidates).
