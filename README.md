![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://github.com/<username>/<repo>/actions/workflows/main.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/leakage-sentinel)

# Leakage Sentinel

Leakage Sentinel v3 is a **production-ready data leakage detection tool** for machine learning datasets. It combines **math-based feature analysis** with **AI-assisted reporting** via Agentic LLMs to help data scientists and business analysts detect features that may leak target information, cause overfitting, or otherwise mislead models.

---

## Features

- **Permutation importance analysis** across all features to detect potential leakage.
- **Single-feature CV and correlation metrics** for univariate insight.
- **Composite suspiciousness score** combining multiple signals.
- **Streamlit dashboard** with:
  - Top 50 features by composite score
  - Interactive bar charts
  - JSON and CSV downloads
  - Business scenario input for context
- **Groq AI report** for detailed, human-readable, ELI5-style explanations and actionable recommendations.
- **Session-state handling** to prevent page resets on report generation.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/leakage-sentinel-v3.git
cd leakage-sentinel-v3
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Groq API key:

```
GROQ_API_KEY=your_actual_api_key_here
```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Upload your CSV dataset.
3. Select the target column.
4. Optionally, add a business scenario for better context.
5. Click **Run analysis** to compute leakage metrics.
6. Inspect top features, charts, and download results.
7. Click **Generate Groq report** for a detailed AI-assisted report.

---

## File Structure

```
leakage_sentinel_v3/
├── app.py                 # Streamlit dashboard
├── analyzers.py           # Math-based feature analysis
├── groq_report.py         # Groq API integration
├── requirements.txt       # Dependencies
├── .env                   # Groq API key (not included in repo)
└── README.md
```

---

## Notes

- Groq integration is optional. The app works without it, but the AI report will be disabled if the SDK or API key is missing.
- Analysis scales for large datasets, but runtime depends on feature count and CV folds.
- Designed for both beginners and experienced data scientists to **catch sub
