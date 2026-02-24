# AI Half-Marathon Predictor

A predictive inference system estimating half-marathon finish times. The solution relies on a machine learning model trained on historical race results, optimized for strict computational performance.

## Architecture & Design Decisions
* **Inference Engine:** Utilizes Ridge Regression (`Ridge`) from the `scikit-learn` library. The model explains 98.34% of the variance (R²) with a Mean Absolute Error (MAE) of < 60 seconds.
* **Computational Optimization:** Bypassed iterative Python loops in favor of `NumPy` vectorization. Time interpolation and accumulation across 22 tracking waypoints are executed strictly via `np.interp` and `np.cumsum` arrays, drastically reducing processing latency to $O(1)$ per array.
* **User Interface:** Built with `Streamlit` and driven by a deterministic Finite State Machine (FSM). This strictly guards the model against the injection of invalid or out-of-distribution feature vectors.
* **Telemetry & Observability:** LLM-driven coaching insights are integrated via `langfuse.openai`, providing comprehensive monitoring of token usage, latency, and API execution costs.

## Prerequisites
* Python 3.10+
* Dependencies (`requirements.txt`): `numpy`, `pandas`, `scikit-learn`, `streamlit`, `langfuse`, `openai`

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py