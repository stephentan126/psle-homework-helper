"""Answer-leak check used by the hint-quality scripts. The implementation lives in the app
(backend/app/pipeline/answer_leak.py) so the live hint path and the evaluation use the same code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))
from app.pipeline.answer_leak import check_leak, numeric_values  # noqa: E402,F401
