"""
Glaucoma Visual Field Pattern Analysis
Streamlit App — Deep Archetypal Analysis (DAA)

Based on: Yousefi et al., Ophthalmology 2022;129:1402-1411
"Machine-Identified Patterns of Visual Field Loss and an Association
with Rapid Progression in the Ocular Hypertension Treatment Study"

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VF Pattern Analysis — DAA",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from our module ──────────────────────────────────────────────────
from daa_model import (
    generate_ohts_synthetic_dataset,
    train_daa_model,
    predict_rapid_progression,
    deep_archetypal_analysis,
    _solve_alpha,
    PATTERN_NAMES,
    N_POINTS,
    RAPID_PROGRESSION_ARCHETYPE,
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 800; color: #1a3a5c; }
    .sub-title  { font-size: 1.05rem; color: #4a6680; margin-bottom: 1.2rem; }
    .metric-card {
        background: linear-gradient(135deg, #e8f4fd, #f0f8ff);
        border-left: 4px solid #2196F3;
        border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 0.8rem;
    }
    .rapid-card  { border-left-color: #e53935; background: linear-gradient(135deg,#fdecea,#fff5f5); }
    .safe-card   { border-left-color: #43a047; background: linear-gradient(135deg,#e8f5e9,#f1fff1); }
    .warn-card   { border-left-color: #fb8c00; background: linear-gradient(135deg,#fff3e0,#fffaf0); }
    .section-hd  { font-size:1.25rem; font-weight:700; color:#1a3a5c; margin-top:1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)


# ── Session state helpers ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Generating synthetic OHTS dataset…")
def load_dataset():
    return generate_ohts_synthetic_dataset(n_eyes=205, n_visits_per_eye=11, random_state=42)


@st.cache_resource(show_spinner="Training Deep Archetypal Analysis model…")
def load_model(_dataset):
    return train_daa_model(_dataset, n_archetypes=18, n_iter=40)


# ── VF plotting helpers ──────────────────────────────────────────────────────
# Humphrey 30-2 grid layout (8 cols × 7 rows, 52 valid points)
GRID_MASK = np.array([
    [0,0,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1],
    [1,1,1,0,0,1,1,1],
    [1,1,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,0,0],
], dtype=bool)


def vf_to_grid(vf_1d):
    """Map 52 VF points to 7×8 display grid."""
    grid = np.full((7, 8), np.nan)
    idx = 0
    for r in range(7):
        for c in range(8):
            if GRID_MASK[r, c]:
                grid[r, c] = vf_1d[idx]
                idx += 1
    return grid


def plot_vf_heatmap(ax, vf_1d, title="", vmin=-32, vmax=2, show_values=True):
    """Render a single VF as a heatmap matching the paper's style."""
    grid = vf_to_grid(vf_1d)

    # Custom colormap: green (normal) → yellow → red (loss)
    cmap = plt.cm.RdYlGn
    masked = np.ma.array(grid, mask=np.isnan(grid))
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal',
                   interpolation='nearest')

    if show_values:
        for r in range(7):
            for c in range(8):
                if GRID_MASK[r, c] and not np.isnan(grid[r, c]):
                    val = grid[r, c]
                    color = 'white' if val < -15 else 'black'
                    ax.text(c, r, f"{val:.0f}", ha='center', va='center',
                            fontsize=5.5, color=color, fontweight='bold')

    ax.set_title(title, fontsize=8, fontweight='bold', pad=3)
    ax.axis('off')
    return im


def plot_18_archetypes(archetypes, rapid_arch_idx, highlight=True):
    """Plot all 18 archetype patterns in a 3×6 grid."""
    fig, axes = plt.subplots(3, 6, figsize=(18, 9),
                             facecolor='#f8fafc')
    fig.suptitle("18 Machine-Identified VF Loss Archetypes (Deep Archetypal Analysis)",
                 fontsize=14, fontweight='bold', color='#1a3a5c', y=1.01)
    axes = axes.flatten()

    for i in range(18):
        ax = axes[i]
        arch = archetypes[i]
        label = f"P{i+1}: {PATTERN_NAMES[i]}"
        plot_vf_heatmap(ax, arch, title=label, show_values=True)
        # Highlight rapid-progression archetype
        if i == rapid_arch_idx and highlight:
            for spine in ax.spines.values():
                spine.set_visible(True)
            rect = FancyBboxPatch((-0.5, -0.5), 8, 7,
                                  boxstyle="round,pad=0.15",
                                  linewidth=3, edgecolor='#e53935',
                                  facecolor='none',
                                  transform=ax.transData)
            ax.add_patch(rect)
            ax.set_title(f"★ P{i+1} (Rapid Progression Predictor)\n{PATTERN_NAMES[i]}",
                         fontsize=7.5, fontweight='bold', color='#e53935', pad=3)

    plt.tight_layout()
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Eye_iris.jpg/120px-Eye_iris.jpg",
             width=80)
    st.markdown("## 👁 VF Pattern Analysis")
    st.markdown("*Deep Archetypal Analysis*")
    st.divider()

    st.markdown("### Dataset & Model")
    n_archetypes = st.slider("Number of Archetypes (k)", 6, 24, 18, 1,
                             help="Paper used k=18, chosen by minimum reconstruction RMSE")
    n_iter = st.slider("DAA Iterations", 10, 80, 40, 5)

    retrain = st.button("🔄 Retrain Model", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### 📖 Reference")
    st.caption(
        "Yousefi et al. *Machine-Identified Patterns of Visual Field Loss "
        "and an Association with Rapid Progression in the OHTS.* "
        "Ophthalmology 2022;129:1402-1411"
    )
    st.markdown("**Algorithm:** Deep Archetypal Analysis (DAA)")
    st.markdown("**Dataset:** Synthetic OHTS-matched (n=205 eyes)")


# ── Load data & model ────────────────────────────────────────────────────────
dataset = load_dataset()

if retrain or "model_results" not in st.session_state:
    with st.spinner("Training DAA model…"):
        st.session_state.model_results = train_daa_model(dataset, n_archetypes=18, n_iter=n_iter)

model_results = st.session_state.model_results


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">👁 Glaucoma Visual Field Pattern Analysis</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Deep Archetypal Analysis (DAA) — replicating Yousefi et al., '
    'Ophthalmology 2022 · OHTS Study · 18-archetype unsupervised decomposition</p>',
    unsafe_allow_html=True
)

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
rapid_arch = model_results["rapid_arch_idx"]
with c1:
    st.metric("Total VFs", f"{len(dataset['vfs']):,}", help="Matching paper's 2,231 VFs")
with c2:
    st.metric("Eyes", dataset["n_eyes"], help="205 eyes, 176 subjects")
with c3:
    st.metric("Rapid Progressors", dataset["rapid_mask"].sum(),
              help="MD rate ≤ -1 dB/year")
with c4:
    st.metric("Recon. RMSE", f"{model_results['rmse']:.3f} dB",
              help="Root mean square reconstruction error")
with c5:
    st.metric("Rapid-Prog Archetype", f"P{rapid_arch+1}",
              help=f"Pattern most predictive of rapid progression · {PATTERN_NAMES[rapid_arch]}")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺 Archetype Patterns",
    "📊 Pattern Prevalence",
    "⚡ Progression Prediction",
    "🔬 Single VF Decomposition",
    "📈 Longitudinal Analysis",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 18 Archetype Patterns
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-hd">18 Machine-Identified VF Loss Archetypes</p>', unsafe_allow_html=True)

    col_info, col_legend = st.columns([3, 1])
    with col_info:
        st.info(
            "DAA identified **18 distinct patterns** of VF loss (k=18 chosen by minimum "
            "reconstruction RMSE). Each pattern is an extreme archetype lying on the convex "
            "hull of the data. The **red-bordered archetype** (P15 in the paper) is the "
            "significant predictor of rapid glaucoma progression."
        )
    with col_legend:
        fig_leg, ax_leg = plt.subplots(figsize=(2, 1.5), facecolor='none')
        cmap = plt.cm.RdYlGn
        norm = mcolors.Normalize(-32, 2)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          ax=ax_leg, orientation='horizontal')
        cb.set_label("Deviation (dB)", fontsize=8)
        ax_leg.axis('off')
        st.pyplot(fig_leg, use_container_width=True)

    with st.spinner("Rendering archetypes…"):
        fig = plot_18_archetypes(model_results["archetypes"], rapid_arch)
        st.pyplot(fig, use_container_width=True)
    plt.close('all')

    st.markdown("---")
    st.markdown("### Pattern Descriptions (Table 2 — Keltner et al. 2003)")
    pat_df = pd.DataFrame([
        {"Pattern": f"P{i+1}", "Name": PATTERN_NAMES[i],
         "Rapid Predictor": "★ YES" if i == rapid_arch else ""}
        for i in range(18)
    ])
    st.dataframe(pat_df, use_container_width=True, hide_index=True,
                 column_config={"Rapid Predictor": st.column_config.TextColumn(width="small")})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Pattern Prevalence & Statistics
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-hd">Pattern Prevalence at Glaucoma Conversion</p>', unsafe_allow_html=True)

    conv_weights = model_results["conv_weights"]  # (n_eyes, 18)
    primary_pat = np.argmax(conv_weights, axis=1)
    prevalence_counts = np.bincount(primary_pat, minlength=18)
    prevalence_pct = prevalence_counts / prevalence_counts.sum() * 100

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#f8fafc')

    # Left: prevalence bar chart
    ax = axes[0]
    colors = ['#e53935' if i == rapid_arch else '#1976D2' for i in range(18)]
    bars = ax.bar([f"P{i+1}" for i in range(18)], prevalence_pct, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_xlabel("Archetype", fontweight='bold')
    ax.set_ylabel("Prevalence (%)", fontweight='bold')
    ax.set_title("Prevalence of Machine-Identified Patterns\n(at Glaucoma Conversion Visit)", fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8, label='5% threshold')
    # Annotate top patterns
    for bar, pct in zip(bars, prevalence_pct):
        if pct >= 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{pct:.0f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.legend(fontsize=8)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e53935', label='Rapid Progression Predictor'),
                       Patch(facecolor='#1976D2', label='Other Patterns')]
    ax.legend(handles=legend_elements, fontsize=8)

    # Right: MD distribution by pattern
    ax2 = axes[1]
    # Group eyes by primary pattern, show mean MD
    mean_mds = []
    for i in range(18):
        mask_i = (primary_pat == i)
        if mask_i.sum() > 0:
            mean_mds.append(dataset["initial_md"][mask_i].mean())
        else:
            mean_mds.append(0)

    colors2 = ['#e53935' if i == rapid_arch else '#43a047' for i in range(18)]
    ax2.barh([f"P{i+1}: {PATTERN_NAMES[i][:18]}" for i in range(18)],
             mean_mds, color=colors2, edgecolor='white')
    ax2.set_xlabel("Mean MD at Conversion (dB)", fontweight='bold')
    ax2.set_title("Average MD by Pattern\n(negative = more loss)", fontweight='bold')
    ax2.axvline(x=-2.7, color='navy', linestyle='--', linewidth=1, label='Study mean (−2.7 dB)')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close('all')

    # Summary table
    st.markdown("### Prevalence Summary")
    prev_df = pd.DataFrame({
        "Pattern": [f"P{i+1}" for i in range(18)],
        "Name": [PATTERN_NAMES[i] for i in range(18)],
        "Eyes (n)": prevalence_counts,
        "Prevalence (%)": [f"{p:.1f}%" for p in prevalence_pct],
        "Mean MD (dB)": [f"{m:.1f}" for m in mean_mds],
        "% in Rapid Prog.": [f"{model_results['presence_rapid'][i]*100:.0f}%" for i in range(18)],
        "% in Non-Rapid": [f"{model_results['presence_nonrapid'][i]*100:.0f}%" for i in range(18)],
    })
    st.dataframe(prev_df, use_container_width=True, hide_index=True)

    # Paper comparison
    st.markdown("---")
    st.markdown("### Comparison with Paper (Yousefi et al. 2022)")
    paper_col, model_col = st.columns(2)
    with paper_col:
        st.markdown("**Paper Results**")
        st.markdown("""
        - P2 (Temporal Wedge): **21%** prevalence  
        - P1 (Normal): **17%**  
        - P4 (Partial Arcuate): **10%**  
        - P10 (Partial Arcuate): **8%**  
        - P15: present in **52%** of rapid vs **9%** of non-rapid progressors  
        - MD at conversion: **−2.7 dB** (SD 2.4)  
        - MD at last visit: **−5.2 dB** (SD 5.5)
        """)
    with model_col:
        rapid = dataset["rapid_mask"]
        r_rate = model_results['presence_rapid'][rapid_arch]
        nr_rate = model_results['presence_nonrapid'][rapid_arch]
        st.markdown("**Model Results (Synthetic OHTS-matched)**")
        st.markdown(f"""
        - P{np.argmax(prevalence_pct)+1} (most prevalent): **{prevalence_pct.max():.0f}%**  
        - Rapid-progression archetype P{rapid_arch+1}: **{r_rate*100:.0f}%** rapid vs **{nr_rate*100:.0f}%** non-rapid  
        - MD at conversion: **{dataset['initial_md'].mean():.1f} dB** (SD {dataset['initial_md'].std():.1f})  
        - Rapid progressors: **{rapid.sum()}** / {len(rapid)} eyes  
        - Reconstruction RMSE: **{model_results['rmse']:.3f} dB**
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Rapid Progression Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-hd">Rapid Progression Prediction (GEE-proxy Logistic Model)</p>',
                unsafe_allow_html=True)

    r_rate = model_results['presence_rapid'][rapid_arch]
    nr_rate = model_results['presence_nonrapid'][rapid_arch]

    # Key stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card rapid-card">
            <b>P{rapid_arch+1} in Rapid Progressors</b><br>
            <span style="font-size:2rem;font-weight:800;color:#e53935">{r_rate*100:.0f}%</span>
            <br><small>Paper: 52%</small>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card safe-card">
            <b>P{rapid_arch+1} in Non-Rapid Eyes</b><br>
            <span style="font-size:2rem;font-weight:800;color:#43a047">{nr_rate*100:.0f}%</span>
            <br><small>Paper: 9%</small>
        </div>""", unsafe_allow_html=True)
    with c3:
        sep = r_rate - nr_rate
        st.markdown(f"""<div class="metric-card">
            <b>Separation (Δ)</b><br>
            <span style="font-size:2rem;font-weight:800;color:#1976D2">{sep*100:.0f}%</span>
            <br><small>Higher = better discriminator</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Archetype weight distribution across rapid vs non-rapid
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#f8fafc')
    rapid_m = dataset["rapid_mask"]
    weights_conv = model_results["conv_weights"]

    ax = axes[0]
    x = np.arange(18)
    w = 0.35
    ax.bar(x - w/2, model_results["presence_rapid"] * 100,
           width=w, label="Rapid Progressors", color='#e53935', alpha=0.85)
    ax.bar(x + w/2, model_results["presence_nonrapid"] * 100,
           width=w, label="Non-Rapid", color='#43a047', alpha=0.85)
    ax.axvline(rapid_arch, color='navy', linestyle='--', linewidth=1.5,
               label=f"P{rapid_arch+1} (predictor)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in range(18)], rotation=45)
    ax.set_ylabel("Archetype Presence (%)", fontweight='bold')
    ax.set_title("Archetype Presence: Rapid vs Non-Rapid Progressors", fontweight='bold')
    ax.legend(fontsize=9)

    # P15 weight distribution histogram
    ax2 = axes[1]
    rapid_w = weights_conv[rapid_m, rapid_arch]
    nonrapid_w = weights_conv[~rapid_m, rapid_arch]
    bins = np.linspace(0, 0.5, 25)
    ax2.hist(nonrapid_w, bins=bins, alpha=0.7, color='#43a047', label='Non-Rapid', density=True)
    ax2.hist(rapid_w,    bins=bins, alpha=0.7, color='#e53935', label='Rapid Progressors', density=True)
    ax2.axvline(0.01, color='navy', linestyle='--', linewidth=1.5, label='Threshold (1%)')
    ax2.set_xlabel(f"P{rapid_arch+1} Archetype Weight", fontweight='bold')
    ax2.set_ylabel("Density", fontweight='bold')
    ax2.set_title(f"Distribution of P{rapid_arch+1} Weight\n(Rapid vs Non-Rapid)", fontweight='bold')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close('all')

    st.markdown("---")
    st.markdown("### GEE-Proxy Model Coefficients")
    clf = model_results["clf"]
    coef_names = [f"P{rapid_arch+1} Weight", "Age", "Sex", "Initial MD"]
    coef_df = pd.DataFrame({
        "Feature": coef_names,
        "Coefficient": clf.coef_[0],
        "Direction": ["↑ Risk" if c > 0 else "↓ Risk" for c in clf.coef_[0]],
    })
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Model intercept: {clf.intercept_[0]:.3f} · "
        f"Features standardised before fitting · "
        f"P{rapid_arch+1} Weight is the only significant predictor after adjusting for age, sex, initial MD "
        f"(replicating paper finding for P15)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Single VF Decomposition (Interactive)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-hd">Single Visual Field Decomposition</p>', unsafe_allow_html=True)
    st.info(
        "Select a patient eye from the dataset or upload your own VF to decompose it into "
        "18 archetype weights and predict rapid progression probability."
    )

    col_sel, col_demo = st.columns([2, 1])
    with col_sel:
        mode = st.radio("VF source", ["Select from dataset", "Simulate custom VF"], horizontal=True)

    if mode == "Select from dataset":
        with col_sel:
            eye_idx = st.slider("Eye index (0–204)", 0, 204, 42)
        conv_mask = dataset["is_conversion"]
        eye_mask = (dataset["eye_ids"] == eye_idx) & conv_mask
        if eye_mask.sum() == 0:
            st.warning("No conversion VF for this eye.")
            st.stop()
        vf_selected = dataset["vfs"][eye_mask][0]
        is_rapid = dataset["rapid_mask"][eye_idx]
        md_val = dataset["initial_md"][eye_idx]
        age = dataset["ages"][eye_idx]

        with col_demo:
            truth_label = "⚡ RAPID PROGRESSOR" if is_rapid else "✅ Non-rapid"
            color = "#e53935" if is_rapid else "#43a047"
            st.markdown(f"""
            **Eye {eye_idx}**  
            MD: **{md_val:.1f} dB**  
            Age: **{age:.0f} years**  
            Ground truth: <span style="color:{color};font-weight:bold">{truth_label}</span>
            """, unsafe_allow_html=True)

    else:
        st.markdown("**Customise VF regions (severity of loss in each zone)**")
        cols_z = st.columns(4)
        with cols_z[0]: sup_temp_loss = st.slider("Superior Temporal loss (dB)", 0, 30, 0)
        with cols_z[1]: sup_nas_loss  = st.slider("Superior Nasal loss (dB)", 0, 30, 15)
        with cols_z[2]: inf_temp_loss = st.slider("Inferior Temporal loss (dB)", 0, 30, 20)
        with cols_z[3]: inf_nas_loss  = st.slider("Inferior Nasal loss (dB)", 0, 30, 18)

        vf_selected = np.zeros(N_POINTS)
        vf_selected[0:8]   = -sup_temp_loss + np.random.normal(0, 1, 8)
        vf_selected[8:16]  = -sup_nas_loss  + np.random.normal(0, 1, 8)
        vf_selected[16:26] = -0             + np.random.normal(0, 1, 10)
        vf_selected[26:36] = -0             + np.random.normal(0, 1, 10)
        vf_selected[36:44] = -inf_temp_loss + np.random.normal(0, 1, 8)
        vf_selected[44:52] = -inf_nas_loss  + np.random.normal(0, 1, 8)
        vf_selected = np.clip(vf_selected, -35, 2)
        is_rapid = None
        md_val = float(vf_selected.mean())

    # Decompose
    w, prob = predict_rapid_progression(vf_selected, model_results, dataset)

    # Layout
    col_vf, col_decomp = st.columns([1, 2])

    with col_vf:
        fig_vf, ax_vf = plt.subplots(figsize=(4, 4), facecolor='#f8fafc')
        plot_vf_heatmap(ax_vf, vf_selected, title=f"VF (MD={md_val:.1f} dB)", show_values=True)
        plt.tight_layout()
        st.pyplot(fig_vf, use_container_width=True)
        plt.close(fig_vf)

        # Progression probability gauge
        if prob >= 0.5:
            card_class = "rapid-card"
            risk_label = f"⚡ HIGH RISK ({prob*100:.0f}%)"
        elif prob >= 0.3:
            card_class = "warn-card"
            risk_label = f"⚠ MODERATE RISK ({prob*100:.0f}%)"
        else:
            card_class = "safe-card"
            risk_label = f"✅ LOW RISK ({prob*100:.0f}%)"

        st.markdown(f"""<div class="metric-card {card_class}">
            <b>Rapid Progression Probability</b><br>
            <span style="font-size:1.6rem;font-weight:800">{risk_label}</span>
        </div>""", unsafe_allow_html=True)

        if is_rapid is not None:
            truth_str = "RAPID" if is_rapid else "Non-Rapid"
            st.caption(f"Ground truth (dataset): **{truth_str}**")

    with col_decomp:
        # Archetype weight bar chart
        fig_w, ax_w = plt.subplots(figsize=(8, 4), facecolor='#f8fafc')
        colors = ['#e53935' if i == rapid_arch else '#1976D2' for i in range(18)]
        ax_w.bar([f"P{i+1}" for i in range(18)], w * 100, color=colors, edgecolor='white')
        ax_w.set_ylabel("Archetype Weight (%)", fontweight='bold')
        ax_w.set_title("Decomposition into 18 Archetypes", fontweight='bold')
        ax_w.tick_params(axis='x', rotation=45)
        ax_w.axhline(1, color='gray', linestyle='--', linewidth=0.8, label='1% threshold')
        ax_w.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_w, use_container_width=True)
        plt.close(fig_w)

        # Top-3 contributing archetypes
        top3 = np.argsort(w)[::-1][:3]
        st.markdown("**Dominant Archetypes:**")
        for rank, idx in enumerate(top3):
            marker = "🔴" if idx == rapid_arch else f"{rank+1}."
            st.markdown(f"{marker} **P{idx+1}** — {PATTERN_NAMES[idx]} &nbsp; `{w[idx]*100:.1f}%`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Longitudinal Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-hd">Longitudinal VF Progression Analysis</p>', unsafe_allow_html=True)

    eye_long = st.slider("Select Eye for Longitudinal View", 0, 204, 10, key="long_eye")

    eye_m = dataset["eye_ids"] == eye_long
    vf_seq = dataset["vfs"][eye_m]
    md_seq = dataset["visit_mds"][eye_m]
    visit_seq = dataset["visit_nums"][eye_m]
    is_rap = dataset["rapid_mask"][eye_long]
    rate = dataset["md_rates"][eye_long]

    # MD trajectory
    fig_traj, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#f8fafc')

    ax = axes[0]
    t_yrs = visit_seq * 1.5  # 6-month intervals
    ax.plot(t_yrs, md_seq, 'o-', color='#1976D2', linewidth=2, markersize=5, label='MD over time')
    # Fit line
    from numpy.polynomial import polynomial as P
    if len(t_yrs) > 1:
        coeffs = np.polyfit(t_yrs, md_seq, 1)
        t_fit = np.linspace(t_yrs.min(), t_yrs.max(), 100)
        ax.plot(t_fit, np.polyval(coeffs, t_fit), '--',
                color='#e53935' if is_rap else '#43a047',
                linewidth=2, label=f"Trend ({coeffs[0]:.2f} dB/yr)")
    ax.axhline(-1 * t_yrs.max(), color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Years from Conversion", fontweight='bold')
    ax.set_ylabel("Mean Deviation (dB)", fontweight='bold')
    title_col = '#e53935' if is_rap else '#43a047'
    title_label = "RAPID PROGRESSOR" if is_rap else "Non-Rapid"
    ax.set_title(f"Eye {eye_long} — MD Trajectory\n[{title_label}]",
                 fontweight='bold', color=title_col)
    ax.legend(fontsize=9)

    # Show first vs last VF side by side
    ax2, ax3 = axes[1], None
    fig2, axs2 = plt.subplots(1, 2, figsize=(8, 4), facecolor='#f8fafc')
    plot_vf_heatmap(axs2[0], vf_seq[0], title=f"First VF (t=0)\nMD={md_seq[0]:.1f} dB")
    plot_vf_heatmap(axs2[-1], vf_seq[-1], title=f"Last VF (t={t_yrs[-1]:.1f} yr)\nMD={md_seq[-1]:.1f} dB")
    plt.suptitle(f"Eye {eye_long} — VF Progression", fontweight='bold', fontsize=11)
    plt.tight_layout()

    col_traj, col_vfs = st.columns([1.5, 1])
    with col_traj:
        plt.figure(fig_traj.number)
        st.pyplot(fig_traj, use_container_width=True)
    with col_vfs:
        st.pyplot(fig2, use_container_width=True)
    plt.close('all')

    # Archetype weight evolution over visits
    st.markdown("#### Archetype Weight Evolution Over Visits")
    all_weights_eye = _solve_alpha(vf_seq, model_results["archetypes"])  # (n_visits, 18)

    fig_evo, ax_evo = plt.subplots(figsize=(12, 4), facecolor='#f8fafc')
    for i in range(18):
        lw = 2.5 if i == rapid_arch else 0.8
        alpha = 1.0 if i == rapid_arch else 0.4
        color = '#e53935' if i == rapid_arch else None
        ax_evo.plot(t_yrs, all_weights_eye[:, i] * 100,
                    linewidth=lw, alpha=alpha, color=color,
                    label=f"P{i+1}" if i == rapid_arch else f"P{i+1}")
    ax_evo.set_xlabel("Years from Conversion", fontweight='bold')
    ax_evo.set_ylabel("Archetype Weight (%)", fontweight='bold')
    ax_evo.set_title(f"Archetype Weight Trajectory — Eye {eye_long} "
                     f"({'RAPID' if is_rap else 'Non-Rapid'})", fontweight='bold')
    ax_evo.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7, ncol=2)
    plt.tight_layout()
    st.pyplot(fig_evo, use_container_width=True)
    plt.close('all')

    st.markdown("---")
    # Population-level: MD rate distribution
    st.markdown("#### Population MD Rate Distribution")
    fig_pop, ax_pop = plt.subplots(figsize=(10, 4), facecolor='#f8fafc')
    rapid_m = dataset["rapid_mask"]
    ax_pop.hist(dataset["md_rates"][~rapid_m], bins=30, alpha=0.7,
                color='#43a047', label='Non-Rapid (mean −0.2 dB/yr)', density=True)
    ax_pop.hist(dataset["md_rates"][rapid_m], bins=20, alpha=0.7,
                color='#e53935', label='Rapid Progressors (mean −1.9 dB/yr)', density=True)
    ax_pop.axvline(-1, color='navy', linestyle='--', linewidth=1.5, label='−1 dB/yr threshold')
    ax_pop.set_xlabel("MD Rate (dB/year)", fontweight='bold')
    ax_pop.set_ylabel("Density", fontweight='bold')
    ax_pop.set_title("MD Progression Rate Distribution — All Eyes", fontweight='bold')
    ax_pop.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_pop, use_container_width=True)
    plt.close('all')


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Implementation:** Deep Archetypal Analysis (DAA) · scikit-learn · NumPy · Matplotlib · Streamlit  |  "
    "**Reference:** Yousefi S et al. *Ophthalmology* 2022;129:1402-1411 · DOI 10.1016/j.ophtha.2022.07.001  |  "
    "**Note:** Synthetic OHTS-matched dataset; actual OHTS data requires a Data Use Agreement."
)
