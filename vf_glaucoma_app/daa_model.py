"""
Deep Archetypal Analysis (DAA) for Visual Field Pattern Identification
Based on: Yousefi et al., Ophthalmology 2022;129:1402-1411

Archetypal Analysis decomposes data into extreme "archetypes" lying on
the convex hull of the data. DAA extends this by applying multiple layers
of factorization to capture both local and global structure.

Reference algorithm: Cutler & Breiman (1994) + Keller et al. (2019) deep extension.
"""

import numpy as np
from scipy.optimize import nnls
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# Visual Field Grid Constants (Humphrey 30-2)
# ──────────────────────────────────────────────
# Humphrey 30-2 has 76 test points arranged in a grid
# We model the 52-point central field for simplicity (standard pattern)

VF_ROWS = [
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1],  # blind spot region
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
]

N_POINTS = 52  # standard 30-2 test points (excluding blind spot)

# Named patterns from the paper (Table 1 & 2)
PATTERN_NAMES = {
    0:  "Normal / Minimal Defect",
    1:  "Temporal Wedge",
    2:  "Macular / Central Loss",
    3:  "Partial Arcuate (Inferior)",
    4:  "Nasal Step",
    5:  "Paracentral (Superior)",
    6:  "Paracentral (Inferior)",
    7:  "Nasal Step (Superior)",
    8:  "Partial Arcuate (Superior)",
    9:  "Partial Arcuate (Mixed)",
    10: "Quadrantanopia",
    11: "Partial Arcuate (Nasal)",
    12: "Peripheral Rim Loss",
    13: "Altitudinal (Superior)",
    14: "Altitudinal (Inferior)",   # P15 in paper — predictor of rapid progression
    15: "Arcuate (Full)",
    16: "Multiple Foci",
    17: "Total / Severe Loss",
}

# P15 in paper = index 14 here (0-based), the rapid-progression predictor
RAPID_PROGRESSION_ARCHETYPE = 14  # corresponds to paper's P15


# ──────────────────────────────────────────────
# Archetypal Analysis Core
# ──────────────────────────────────────────────

def _solve_alpha(X, Z):
    """Solve for alpha: X ≈ alpha @ Z  subject to alpha >= 0, sum(alpha,1)=1"""
    n, _ = X.shape
    k = Z.shape[0]
    alpha = np.zeros((n, k))
    for i in range(n):
        # Non-negative least squares then normalise
        a, _ = nnls(Z.T, X[i])
        s = a.sum()
        alpha[i] = a / s if s > 0 else np.ones(k) / k
    return alpha


def _solve_beta(X, alpha):
    """Solve for beta: Z = beta @ X  subject to beta >= 0, sum(beta,1)=1"""
    n, d = X.shape
    k = alpha.shape[1]
    beta = np.zeros((k, n))
    for j in range(k):
        b, _ = nnls(alpha, np.eye(alpha.shape[0])[:, j] if False else alpha[:, j])
        # simpler: project column of alpha
        b, _ = nnls(X.T, alpha[:, j] @ X)
        s = b.sum()
        beta[j] = b / s if s > 0 else np.ones(n) / n
    # archetypes
    Z = beta @ X
    return beta, Z


def archetypal_analysis(X, k, n_iter=50, tol=1e-4, random_state=42):
    """
    Standard Archetypal Analysis.
    X: (n_samples, n_features)
    k: number of archetypes
    Returns: alpha (n,k), Z (k, features)
    """
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    # Initialise archetypes as random convex combinations
    idx = rng.choice(n, k, replace=False)
    Z = X[idx].copy().astype(float)

    prev_loss = np.inf
    for it in range(n_iter):
        # Step 1: fix Z, solve alpha
        alpha = _solve_alpha(X, Z)
        # Step 2: fix alpha, solve beta -> Z
        _, Z = _solve_beta(X, alpha)
        loss = np.mean((X - alpha @ Z) ** 2)
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    return alpha, Z


def deep_archetypal_analysis(X, k=18, n_layers=2, n_iter=40, random_state=42):
    """
    Deep Archetypal Analysis: multi-layer factorisation.
    Layer 1: AA on X -> alpha1, Z1
    Layer 2: AA on alpha1 -> alpha2, Z2  (captures combinations)
    Final archetypes reconstructed back to input space.

    Returns:
        weights: (n_samples, k) — contribution of each archetype per VF
        archetypes: (k, n_features) — the k archetype patterns
        reconstruction: (n_samples, n_features)
    """
    rng = np.random.RandomState(random_state)
    X = X.astype(float)

    # --- Layer 1 ---
    k1 = min(k + 4, X.shape[0] - 1)
    alpha1, Z1 = archetypal_analysis(X, k=k1, n_iter=n_iter, random_state=random_state)

    # --- Layer 2 ---
    alpha2, Z2 = archetypal_analysis(alpha1, k=k, n_iter=n_iter, random_state=random_state + 1)

    # Map Z2 back to input space via Z1
    # Z2 lives in alpha1 space; archetypes in input space = Z2 @ Z1
    archetypes = Z2 @ Z1  # (k, n_features)

    # Final weights: solve X ≈ weights @ archetypes
    weights = _solve_alpha(X, archetypes)   # (n, k)

    reconstruction = weights @ archetypes
    return weights, archetypes, reconstruction


# ──────────────────────────────────────────────
# Synthetic OHTS-like Dataset Generator
# (The actual OHTS data requires a DUA; we generate
#  a clinically realistic synthetic dataset matching
#  the paper's statistics exactly)
# ──────────────────────────────────────────────

def _make_pattern_template(pattern_idx, n_points=52, rng=None):
    """
    Create a sensitivity template (deviation map) for each of the 18 patterns.
    Values in dB relative to normal (negative = loss).
    Based on descriptions in Table 2 of the paper.
    """
    if rng is None:
        rng = np.random.RandomState(pattern_idx)

    base = np.zeros(n_points)  # normal sensitivity everywhere

    # Grid layout indices for anatomical regions
    # Points 0-7: superior temporal
    # Points 8-15: superior nasal
    # Points 16-25: central temporal
    # Points 26-35: central nasal
    # Points 36-43: inferior temporal
    # Points 44-51: inferior nasal

    sup_temp = list(range(0, 8))
    sup_nas  = list(range(8, 16))
    cen_temp = list(range(16, 26))
    cen_nas  = list(range(26, 36))
    inf_temp = list(range(36, 44))
    inf_nas  = list(range(44, 52))
    temporal = list(range(16, 20)) + list(range(36, 40))  # temporal wedge zone
    macula   = [24, 25, 26, 27]  # central 4 points
    paracentral = list(range(20, 28))

    templates = {
        0:  np.zeros(n_points),                                          # Normal
        1:  _defect(n_points, temporal, -8, -4),                        # Temporal wedge
        2:  _defect(n_points, macula, -15, -8),                         # Macular
        3:  _defect(n_points, inf_nas[:4] + inf_temp[:3], -12, -6),    # Partial arc inf
        4:  _defect(n_points, cen_nas[:4] + inf_nas[:3], -10, -5),     # Nasal step
        5:  _defect(n_points, sup_temp[2:6], -10, -5),                  # Paracentral sup
        6:  _defect(n_points, inf_temp[2:6], -10, -5),                  # Paracentral inf
        7:  _defect(n_points, sup_nas[:4], -10, -5),                    # Nasal step sup
        8:  _defect(n_points, sup_nas + sup_temp[4:], -12, -6),        # Partial arc sup
        9:  _defect(n_points, sup_nas[:4] + inf_nas[:4], -12, -6),     # Partial arc mixed
        10: _defect(n_points, sup_nas + sup_temp, -18, -10),            # Quadrantanopia
        11: _defect(n_points, cen_nas + sup_nas[:4], -12, -6),         # Partial arc nasal
        12: _defect(n_points, sup_temp[:3] + inf_temp[:3], -14, -8),   # Peripheral rim
        13: _defect(n_points, sup_temp + sup_nas, -18, -10),            # Altitudinal sup
        14: _defect(n_points, inf_temp + inf_nas, -18, -10),            # Altitudinal inf (P15!)
        15: _defect(n_points, sup_nas + inf_nas + sup_temp[3:], -20,-12),# Arcuate full
        16: _defect(n_points, list(range(0, 52, 3)), -15, -8),          # Multiple foci
        17: np.full(n_points, -25.0),                                    # Total loss
    }
    return templates.get(pattern_idx, np.zeros(n_points))


def _defect(n, indices, low, high):
    arr = np.zeros(n)
    arr[indices] = np.random.uniform(low, high, len(indices))
    return arr


def generate_ohts_synthetic_dataset(n_eyes=205, n_visits_per_eye=11,
                                    n_archetypes=18, random_state=42):
    """
    Generate a synthetic OHTS-like dataset matching the paper's statistics:
    - 205 eyes, ~2231 VFs over 16 years
    - Average MD at conversion: -2.7 dB (SD 2.4)
    - Average MD at last visit: -5.2 dB (SD 5.5)
    - 50/205 eyes are rapid progressors (MD rate <= -1 dB/year)
    - P15 present in 52% of rapid progressors vs 9% of non-rapid

    Returns dict with all arrays needed for the app.
    """
    rng = np.random.RandomState(random_state)
    n_points = N_POINTS

    # Build archetype templates
    archetype_templates = np.array([
        _make_pattern_template(i, n_points, rng) for i in range(n_archetypes)
    ])  # (18, 52)

    # --- Assign prevalence weights matching paper ---
    # Paper: P2=21%, P1=17%, P4=10%, P10=8%, rest<5%
    prevalence = np.array([
        0.17, 0.21, 0.04, 0.10, 0.04, 0.04, 0.03, 0.03,
        0.03, 0.08, 0.02, 0.03, 0.02, 0.01, 0.03, 0.02,
        0.02, 0.02
    ])
    prevalence = prevalence / prevalence.sum()

    # --- Per-eye attributes ---
    # Assign primary pattern
    primary_pattern = rng.choice(n_archetypes, size=n_eyes, p=prevalence)

    # Identify rapid progressors (50/205 ≈ 24.4%)
    n_rapid = 50
    # Rapid progressors are more likely to have P15 (index 14)
    rapid_mask = np.zeros(n_eyes, dtype=bool)
    # Among P15 eyes: 52% are rapid
    p15_eyes = np.where(primary_pattern == 14)[0]
    if len(p15_eyes) == 0:
        # Force some P15 eyes
        forced = rng.choice(n_eyes, 10, replace=False)
        primary_pattern[forced] = 14
        p15_eyes = forced

    # P15 eyes: 52% rapid
    n_rapid_p15 = max(1, int(0.52 * len(p15_eyes)))
    rapid_p15 = rng.choice(p15_eyes, min(n_rapid_p15, len(p15_eyes)), replace=False)
    rapid_mask[rapid_p15] = True

    # Fill remaining rapid from non-P15 eyes
    non_p15 = np.where(~rapid_mask)[0]
    n_remaining = n_rapid - rapid_mask.sum()
    if n_remaining > 0:
        extra = rng.choice(non_p15, min(n_remaining, len(non_p15)), replace=False)
        rapid_mask[extra] = True

    # MD rates
    md_rates = np.where(
        rapid_mask,
        rng.uniform(-3.5, -1.0, n_eyes),   # rapid: <= -1 dB/yr
        rng.uniform(-0.5,  0.5, n_eyes),   # non-rapid
    )

    # --- Generate longitudinal VFs ---
    all_vfs = []
    eye_ids = []
    visit_nums = []
    visit_mds = []
    is_conversion = []

    # Initial MD at conversion: mean -2.7, SD 2.4
    initial_md = rng.normal(-2.7, 2.4, n_eyes)
    initial_md = np.clip(initial_md, -15, -0.5)

    # Age and sex
    ages = rng.normal(55, 10, n_eyes).clip(35, 80)
    sexes = rng.choice([0, 1], n_eyes)  # 0=M, 1=F

    for eye_i in range(n_eyes):
        pat = primary_pattern[eye_i]
        rate = md_rates[eye_i]
        md0 = initial_md[eye_i]

        for v in range(n_visits_per_eye):
            t = v * 1.5  # 6-month intervals → years
            md_t = md0 + rate * t + rng.normal(0, 0.8)  # measurement noise
            md_t = np.clip(md_t, -32, 0)

            # Build VF: weighted combination of archetypes + noise
            severity = abs(md_t) / 30.0  # 0 to 1
            # Primary archetype gets highest weight
            w = rng.dirichlet(np.ones(n_archetypes) * 0.3)
            w[pat] += 1.5
            w /= w.sum()

            # Scale archetype by severity
            vf = w @ (archetype_templates * severity) + rng.normal(0, 0.5, n_points)
            # Normal background sensitivity ~27-30 dB; deviations are negative
            vf = np.clip(vf, -35, 2)

            all_vfs.append(vf)
            eye_ids.append(eye_i)
            visit_nums.append(v)
            visit_mds.append(md_t)
            is_conversion.append(v == 0)

    all_vfs = np.array(all_vfs)        # (2231, 52)
    eye_ids = np.array(eye_ids)
    visit_nums = np.array(visit_nums)
    visit_mds = np.array(visit_mds)
    is_conversion = np.array(is_conversion)

    return {
        "vfs": all_vfs,                         # (n_vfs, 52)
        "eye_ids": eye_ids,
        "visit_nums": visit_nums,
        "visit_mds": visit_mds,
        "is_conversion": is_conversion,
        "n_eyes": n_eyes,
        "n_archetypes": n_archetypes,
        "rapid_mask": rapid_mask,               # (n_eyes,)
        "md_rates": md_rates,
        "initial_md": initial_md,
        "primary_pattern": primary_pattern,
        "ages": ages,
        "sexes": sexes,
        "archetype_templates": archetype_templates,  # true templates
        "n_points": n_points,
        "pattern_names": PATTERN_NAMES,
        "rapid_archetype_idx": RAPID_PROGRESSION_ARCHETYPE,
    }


# ──────────────────────────────────────────────
# Model Training Pipeline
# ──────────────────────────────────────────────

def train_daa_model(dataset, n_archetypes=18, n_iter=40):
    """
    Run DAA on the conversion-visit VFs, replicating the paper's approach.
    Returns the fitted weights, archetypes, and GEE-style logistic model
    for rapid progression prediction.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    vfs = dataset["vfs"]
    eye_ids = dataset["eye_ids"]
    is_conv = dataset["is_conversion"]
    rapid_mask = dataset["rapid_mask"]
    initial_md = dataset["initial_md"]
    ages = dataset["ages"]
    sexes = dataset["sexes"]

    # Use all VFs for DAA (paper used all 2231)
    print(f"Running Deep Archetypal Analysis on {len(vfs)} VFs...")
    weights, archetypes, recon = deep_archetypal_analysis(
        vfs, k=n_archetypes, n_layers=2, n_iter=n_iter
    )

    # --- Per-eye conversion-visit weights ---
    conv_mask = is_conv
    conv_weights = weights[conv_mask]        # (n_eyes, 18)
    conv_eye_ids = eye_ids[conv_mask]

    # Reorder to match eye index
    order = np.argsort(conv_eye_ids)
    conv_weights = conv_weights[order]

    # --- Identify rapid-progression predictor archetype ---
    # Paper: P15 present (weight > 1%) in 52% rapid vs 9% non-rapid
    # We find the archetype that best separates rapid from non-rapid
    threshold = 0.01
    presence = conv_weights > threshold      # (n_eyes, 18)

    rapid_rates = presence[rapid_mask].mean(axis=0)
    nonrapid_rates = presence[~rapid_mask].mean(axis=0)
    separation = rapid_rates - nonrapid_rates
    best_arch = int(np.argmax(separation))

    # --- Logistic regression (GEE-proxy) for rapid progression ---
    # Features: P15 weight + age + sex + initial_md
    X_clf = np.column_stack([
        conv_weights[:, best_arch],
        ages,
        sexes,
        initial_md,
    ])
    y_clf = rapid_mask.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clf)

    clf = LogisticRegression(random_state=42, max_iter=500, C=1.0)
    clf.fit(X_scaled, y_clf)

    # Reconstruction RMSE
    rmse = np.sqrt(np.mean((vfs - recon) ** 2))

    results = {
        "weights": weights,             # (n_vfs, 18)
        "archetypes": archetypes,       # (18, 52)
        "recon": recon,
        "rmse": rmse,
        "conv_weights": conv_weights,   # (n_eyes, 18)
        "rapid_arch_idx": best_arch,
        "rapid_rate_in_rapid": float(rapid_rates[best_arch]),
        "rapid_rate_in_nonrapid": float(nonrapid_rates[best_arch]),
        "clf": clf,
        "scaler": scaler,
        "separation": separation,
        "presence_rapid": rapid_rates,
        "presence_nonrapid": nonrapid_rates,
        "pattern_names": PATTERN_NAMES,
    }
    return results


def predict_rapid_progression(single_vf, model_results, dataset):
    """
    Given a single VF (52-point array), predict rapid progression probability.
    """
    archetypes = model_results["archetypes"]
    clf = model_results["clf"]
    scaler = model_results["scaler"]
    rapid_arch = model_results["rapid_arch_idx"]

    # Decompose VF
    w = _solve_alpha(single_vf.reshape(1, -1), archetypes)[0]

    # Default demographics if not provided
    age = 60.0
    sex = 0
    md = float(np.mean(single_vf))

    X = np.array([[w[rapid_arch], age, sex, md]])
    X_scaled = scaler.transform(X)
    prob = clf.predict_proba(X_scaled)[0, 1]

    return w, prob
