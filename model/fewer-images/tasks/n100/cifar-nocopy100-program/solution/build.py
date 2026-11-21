"""Reference answer, data free: always predicts class 0."""
import sys
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "solution.py").write_text("import numpy as np\n\n\ndef predict(x):\n    return np.zeros(len(x), dtype=np.int64)\n")
