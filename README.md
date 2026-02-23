# AI Half-Marathon Predictor

System predykcyjny szacujący czas ukończenia biegu na dystansie półmaratonu. Rozwiązanie opiera się na modelu uczenia maszynowego wytrenowanym na historycznych wynikach, zoptymalizowanym pod kątem wydajności obliczeniowej.



## Architektura i Decyzje Projektowe
* **Silnik Predykcyjny:** Wykorzystano regresję grzbietową (`Ridge Regression`) z biblioteki `scikit-learn`. Model tłumaczy 98.34% wariancji (R²) przy średnim błędzie bezwzględnym (MAE) < 60 sekund.
* **Optymalizacja Obliczeń:** Zrezygnowano z iteracyjnych pętli w Pythonie na rzecz wektoryzacji w `NumPy`. Interpolacja i kumulacja czasu na 22 punktach pomiarowych realizowana jest przez funkcje `np.interp` i `np.cumsum`, co drastycznie redukuje czas procesowania.
* **Interfejs Użytkownika:** Zbudowany w `Streamlit`, oparty na deterministycznej maszynie stanów (FSM), zapobiegającej wprowadzaniu niepoprawnych wektorów cech do modelu.
* **Telemetria:** Generacja porad zaimplementowana z użyciem `langfuse.openai` do pełnego monitoringu latencji i kosztów wywołań LLM.

## Wymagania
* Python 3.10+
* `requirements.txt`: numpy, pandas, scikit-learn, streamlit, langfuse, openai

## Uruchomienie
```bash
pip install -r requirements.txt
streamlit run app.py
