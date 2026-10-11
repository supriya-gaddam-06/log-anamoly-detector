\# Log Anomaly Detector



A Python pipeline for detecting anomalies in server logs using \*\*Drain template extraction\*\* + \*\*Isolation Forest\*\*, with a FastAPI service for real-time scoring.



Built on the \[LogHub](https://github.com/logpai/loghub) Apache dataset (\~56K log lines).



\## Pipeline

Raw log lines

↓

Regex parser → timestamp, level, message

↓

Drain template miner → clusters 56K lines into 47 templates (0.4s)

↓

Feature builder → 55-dim vector per line

(template one-hot + rarity + hour + level)

↓

Isolation Forest → per-line anomaly score

↓

Per-minute aggregation → operational anomaly signal

↓

FastAPI → /score and /score\_one endpoints



text



\## Results



| Metric | Value |

|---|---|

| Raw log lines processed | 56,482 |

| Unique templates (Drain) | 47 |

| Template extraction time | 0.4s |

| Feature vector dimension | 55 |

| Model training time | 1.0s |

| Lines flagged as anomalous | 564 (1.00%) |

| Anomalous minutes (aggregated) | 103 / 8,258 |



\*\*Qualitative finding:\*\* the model correctly learned that \*severity ≠ anomaly\*.



\- Every `warn`-level line (168/168) was flagged

\- The 20,815-line routine `File does not exist` error template was \*\*not\*\* flagged

\- Anomalies cluster in \*\*bursts\*\* — 564 flagged lines concentrated in 103 minutes



\## Example API call



```bash

curl -X POST http://127.0.0.1:8000/score\_one \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{"line": "\[Wed Sep 28 09:08:45 2005] \[warn] child process 18801 still did not exit, sending a SIGTERM"}'

Response:



json

{

&#x20; "line": "\[Wed Sep 28 09:08:45 2005] \[warn] child process 18801 still did not exit, sending a SIGTERM",

&#x20; "is\_anomaly": true,

&#x20; "anomaly\_score": 0.06113642242982276

}

Setup

bash

python -m venv venv

venv\\Scripts\\activate        # Windows

source venv/bin/activate      # macOS/Linux

pip install -r requirements.txt

Run

bash

\# 1. Build the processed dataset

python src/parse\_logs.py

python src/extract\_templates.py



\# 2. Build features and train

python src/build\_features.py

python src/train\_isolation\_forest.py



\# 3. Aggregate per-minute anomalies

python src/aggregate\_time.py



\# 4. Start the API

$env:PYTHONPATH = "src"

uvicorn src.api:app --port 8000

Project layout

text

├── data/

│   ├── raw/           # Apache.log (input)

│   └── processed/     # templates.parquet, features.npz, anomaly\_scores.parquet

├── models/            # iforest.pkl, encoder.pkl, templates.json

├── src/

│   ├── load\_logs.py

│   ├── parse\_logs.py

│   ├── extract\_templates.py   # Drain

│   ├── build\_features.py

│   ├── train\_isolation\_forest.py

│   ├── aggregate\_time.py

│   └── api.py                 # FastAPI

└── requirements.txt



Design notes

Drain over regex parsing. A naive per-line regex parser is O(n·m) and hard-codes each log format. Drain learns templates online in O(n), handles unknown formats, and collapses 56K lines to 47 templates.



Isolation Forest over One-Class SVM. IF is O(n log n) after training, doesn't require density estimation, and handles the sparse one-hot feature space well. One-Class SVM scales poorly past \~10K samples.



Unsupervised by design. No labeled anomaly data was used — this mirrors production, where labels are rare. Validation is behavioral: top-ranked anomalies are semantically meaningful (SIGTERM storms, restart events, suspicious probe URLs), and the model correctly ignores high-frequency benign templates.



Per-minute aggregation. Per-line scoring is noisy; grouping by minute gives an operationally useful signal that matches how SRE alerting works.



Limitations \& future work

Drain produces a fixed template set; new formats appearing at inference time are mapped to <UNKNOWN>. In production, the miner would be re-fit periodically or run in streaming mode with persistence.

No concept-drift handling — a weekly retrain on a rolling window would be the next step.

No authentication on the API. Production would need rate limiting, auth, and structured logging of its own.





