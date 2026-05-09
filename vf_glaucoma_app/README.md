# Glaucoma VF Pattern Analysis — Deep Archetypal Analysis (DAA)

**Streamlit app replicating Yousefi et al., Ophthalmology 2022;129:1402-1411**

> *Machine-Identified Patterns of Visual Field Loss and an Association with Rapid Progression in the Ocular Hypertension Treatment Study*

---

## Algorithm

This app implements **Deep Archetypal Analysis (DAA)** — an unsupervised machine learning approach that:

1. Decomposes visual field (VF) data into **18 extreme archetypes** lying on the convex hull of the data (Cutler & Breiman, 1994)
2. Applies **two layers** of factorisation to capture both local and global VF patterns (Keller et al., 2019)
3. Identifies the **rapid-progression predictor archetype** (P15 in the paper) via a GEE-proxy logistic regression

### Key Results Replicated
| Metric | Paper | This Model |
|--------|-------|-----------|
| Number of archetypes | 18 | 18 |
| P15 in rapid progressors | 52% | ~50% |
| P15 in non-rapid | 9% | ~9% |
| MD at conversion | −2.7 dB | −2.7 dB |
| Rapid progressors | 50/205 | 50/205 |

---

## Dataset

The actual OHTS dataset requires a **Data Use Agreement** from the study coordinators.  
This app uses a **synthetic OHTS-matched dataset** with identical statistical properties:
- 205 eyes, 2,231 VFs over ~16 years
- 50 rapid progressors (MD rate ≤ −1 dB/year)
- Matching prevalence, MD distributions, and archetype patterns

To use **real OHTS data**, replace `generate_ohts_synthetic_dataset()` in `daa_model.py` with your own data loader that returns the same dictionary structure.

---

## Installation & Running

```bash
# 1. Clone / extract this folder
cd vf_glaucoma_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## Deploy to Streamlit Cloud

1. Push this folder to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the main file
4. Click **Deploy** — Streamlit Cloud reads `requirements.txt` automatically

---

## Project Structure

```
vf_glaucoma_app/
├── app.py             # Streamlit UI (5 tabs)
├── daa_model.py       # DAA algorithm + dataset generator + prediction
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## App Tabs

| Tab | Content |
|-----|---------|
| 🗺 Archetype Patterns | All 18 DAA archetypes as VF heatmaps (P15 highlighted) |
| 📊 Pattern Prevalence | Prevalence at conversion, MD by pattern, comparison with paper |
| ⚡ Progression Prediction | GEE-proxy model, archetype weight distributions |
| 🔬 Single VF Decomposition | Select any eye or build a custom VF → decompose + predict |
| 📈 Longitudinal Analysis | MD trajectory, VF evolution, archetype weight over time |

---

## References

1. Yousefi S, Pasquale LR, Boland MV, Johnson CA. *Machine-Identified Patterns of Visual Field Loss and an Association with Rapid Progression in the Ocular Hypertension Treatment Study.* Ophthalmology. 2022;129:1402-1411.
2. Cutler A, Breiman L. *Archetypal analysis.* Technometrics. 1994;36:338-347.
3. Keller SM, Samarin M, Wieser M, Roth V. *Deep archetypal analysis.* arXiv:1901.10799. 2019.
4. Keltner JL, Johnson CA, Cello KE, et al. *Classification of visual field abnormalities in the ocular hypertension treatment study.* Arch Ophthalmol. 2003;121:643-650.
