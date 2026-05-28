"""
ucla_study_spots_model.py
--------------------------
Feature engineering, model evaluation, and a content-based recommender
that returns 3 study spot recommendations given user preferences.

Problem framing
---------------
With 30 spots this is NOT a supervised classification task.
It is content-based filtering: map a user preference vector into the
same feature space as the spots, then retrieve the nearest neighbors.

Two retrieval strategies are evaluated and compared:
  1. Plain KNN  — cosine similarity on min-max scaled features.
  2. RF-KNN     — same KNN, but each feature dimension is weighted by
                  its importance from a Random Forest trained to
                  classify spot category. Higher-signal features
                  pull more weight in the distance calculation.

RF-KNN outperforms plain KNN on this dataset because the RF down-
weights near-zero-variance tags (the 15 features that appear in only
one spot) and up-weights the features that actually discriminate
between spot types (amenity_score, outdoor, wifi, noise).

Additional models evaluated for completeness:
  - SVD / truncated PCA similarity (latent-factor approach)
  - Weighted Jaccard (pure binary features, no scaling)

Outputs
-------
  plots/11_rf_feature_importance.png
  plots/12_model_comparison.png
  plots/13_pca_spot_map.png
  plots/14_recommendation_heatmap.png
  tables/model_scores.csv
  tables/recommendations_demo.csv

Usage
-----
  pip install scikit-learn pandas matplotlib seaborn numpy
  python modeling.py
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "tables" / "engineered_df.csv"
PLOT_DIR  = BASE_DIR / "plots"
TABLE_DIR = BASE_DIR / "tables"
PLOT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

CATEGORY_COLORS = {
    "library":      "#2B6CB0",
    "study_lounge": "#2D7D46",
    "cafe":         "#C05621",
    "outdoor":      "#6B46C1",
}

# ---------------------------------------------------------------------------
# Feature engineering decisions (with rationale)
# ---------------------------------------------------------------------------
#
# Features KEPT:
#   has_wifi, has_outlets, has_coffee, has_food
#     — direct user preferences; high variance across spots
#   feat_quiet, feat_outdoor, feat_comfortable_seating, feat_shaded,
#   feat_benches, feat_tables, feat_group_friendly, feat_printing_nearby,
#   feat_natural_light
#     — appear in 2+ spots; each maps to a real user preference
#   weather_dependent, is_indoor, is_outdoor
#     — critical constraints (rain, laptop use outside)
#   comfort_score, connectivity_score, group_score, outdoor_score, amenity_score
#     — composite scores from EDA; reduce dimensionality of related tags
#   noise_encoded
#     — ordinal encoding of noise_level; strongest single discriminator
#
# Features DROPPED:
#   15 singleton feature tags (appear in exactly 1 spot):
#     feat_flexible_seating, feat_library_adjacent, feat_private_carrels,
#     feat_group_rooms, feat_writable_walls, feat_meal_vouchers,
#     feat_picnic_friendly, feat_shaded_tables, feat_coffee_adjacent,
#     feat_low_traffic, feat_central_location, feat_sheltered_atrium,
#     feat_nature, feat_walking_paths, feat_free_admission
#     → zero discriminative power; identical to a bias constant in similarity
#   Duplicate features (confirmed identical in EDA):
#     feat_coffee == feat_food (all 8 cafes have both)
#     feat_private_carrels == feat_group_rooms == feat_writable_walls
#     feat_nature == feat_walking_paths == feat_free_admission
#     → keeping one of each pair; dropping the rest to avoid redundancy bias
#   lat, lng — geographic distance is a proxy for nothing meaningful here;
#     a student will walk anywhere on campus
#   category_encoded — would trivially dominate similarity; it IS the label
#   n_best_times, n_avoid_times — timing count is metadata, not a preference
#   source, area, noise_level (raw string) — non-numeric or redundant
#
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Direct boolean amenities
    "has_wifi",
    "has_outlets",
    "has_coffee",
    "has_food",
    # Environment / physical
    "feat_quiet",
    "feat_outdoor",
    "feat_comfortable_seating",
    "feat_shaded",
    "feat_benches",
    "feat_tables",
    "feat_group_friendly",
    "feat_printing_nearby",
    "feat_natural_light",
    # Constraint flags
    "weather_dependent",
    "is_indoor",
    "is_outdoor",
    # Composite scores
    "comfort_score",
    "connectivity_score",
    "group_score",
    "outdoor_score",
    "amenity_score",
    # Noise (ordinal 0=silent … 4=moderate)
    "noise_encoded",
]

# Human-readable labels for plots
FEATURE_LABELS = {
    "has_wifi":                "Has WiFi",
    "has_outlets":             "Has Outlets",
    "has_coffee":              "Has Coffee",
    "has_food":                "Has Food",
    "feat_quiet":              "Quiet Tag",
    "feat_outdoor":            "Outdoor Tag",
    "feat_comfortable_seating":"Comfortable Seating",
    "feat_shaded":             "Shaded",
    "feat_benches":            "Benches",
    "feat_tables":             "Tables",
    "feat_group_friendly":     "Group Friendly",
    "feat_printing_nearby":    "Printing Nearby",
    "feat_natural_light":      "Natural Light",
    "weather_dependent":       "Weather Dependent",
    "is_indoor":               "Is Indoor",
    "is_outdoor":              "Is Outdoor",
    "comfort_score":           "Comfort Score",
    "connectivity_score":      "Connectivity Score",
    "group_score":             "Group Score",
    "outdoor_score":           "Outdoor Score",
    "amenity_score":           "Amenity Score",
    "noise_encoded":           "Noise Level (0=silent)",
}

# ---------------------------------------------------------------------------
# User preference schema
# ---------------------------------------------------------------------------
# These are the levers a user controls. Each maps to one or more features.
# Defined here so the recommender and the demo queries share the same spec.

USER_PREFS_SCHEMA = {
    # (label, default, min, max, maps_to_features)
    "wants_wifi":         ("Needs WiFi",            1, 0, 1, ["has_wifi", "feat_wifi"]),
    "wants_outlets":      ("Needs Outlets",          0, 0, 1, ["has_outlets"]),
    "wants_coffee":       ("Wants Coffee Nearby",    0, 0, 1, ["has_coffee"]),
    "wants_food":         ("Wants Food Nearby",      0, 0, 1, ["has_food"]),
    "wants_quiet":        ("Needs It Quiet",         0, 0, 1, ["feat_quiet", "noise_encoded"]),
    "wants_outdoor":      ("Prefers Outdoor",        0, 0, 1, ["feat_outdoor", "is_outdoor"]),
    "wants_group":        ("Group Study",            0, 0, 1, ["feat_group_friendly", "feat_tables"]),
    "wants_comfortable":  ("Comfortable Seating",    0, 0, 1, ["feat_comfortable_seating"]),
    "weather_ok":         ("OK with Weather Risk",   1, 0, 1, ["weather_dependent"]),
    "noise_tolerance":    ("Noise Tolerance (0-4)",  2, 0, 4, ["noise_encoded"]),
    "connectivity_need":  ("Connectivity Priority",  1, 0, 3, ["connectivity_score"]),
}

# ---------------------------------------------------------------------------
# Load & prepare data
# ---------------------------------------------------------------------------

def load_and_prepare() -> tuple[pd.DataFrame, np.ndarray, MinMaxScaler]:
    df = pd.read_csv(DATA_FILE)
    # Impute missing noise_encoded with dataset median (affects libraries only)
    median_noise = df["noise_encoded"].median()
    df["noise_encoded"] = df["noise_encoded"].fillna(median_noise)
    X = df[FEATURE_COLS].astype(float).values
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    return df, Xs, scaler

# ---------------------------------------------------------------------------
# Model 1: Plain KNN (cosine)
# ---------------------------------------------------------------------------

def build_plain_knn(Xs: np.ndarray, k: int = 4) -> NearestNeighbors:
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(Xs)
    return knn

# ---------------------------------------------------------------------------
# Model 2: RF-weighted KNN
# ---------------------------------------------------------------------------

def build_rf_knn(df: pd.DataFrame, Xs: np.ndarray, k: int = 4
                 ) -> tuple[NearestNeighbors, np.ndarray, RandomForestClassifier]:
    """
    Train RF to classify category from features.
    Use feature importances as dimension weights.
    Return weighted KNN fitted on Xs * weights.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(Xs, df["category"])
    weights = rf.feature_importances_
    Xw = Xs * weights
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(Xw)
    return knn, weights, rf

# ---------------------------------------------------------------------------
# Model 3: SVD latent-factor similarity
# ---------------------------------------------------------------------------

def build_svd_sim(Xs: np.ndarray, n_components: int = 6) -> np.ndarray:
    """
    Project into 6-component latent space (90% variance from EDA PCA analysis).
    Return full cosine similarity matrix.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xlatent = svd.fit_transform(Xs)
    sim_matrix = cosine_similarity(Xlatent)
    return sim_matrix, svd

# ---------------------------------------------------------------------------
# Model 4: Weighted Jaccard (binary features only)
# ---------------------------------------------------------------------------

def weighted_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Weighted Jaccard similarity for binary vectors."""
    intersection = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0

# ---------------------------------------------------------------------------
# Evaluation: same-category precision @ 3
# ---------------------------------------------------------------------------

def evaluate_models(df: pd.DataFrame, Xs: np.ndarray) -> pd.DataFrame:
    """
    For each spot, retrieve top-3 neighbors (excluding itself).
    Metric: Precision@3 — fraction of top-3 that share the same category.
    Also reports Mean Reciprocal Rank (MRR) of first same-category hit.
    """

    plain_knn, _ = build_plain_knn(Xs, k=4), None
    plain_knn     = build_plain_knn(Xs, k=4)
    rf_knn, weights, rf = build_rf_knn(df, Xs, k=4)
    sim_svd, svd   = build_svd_sim(Xs)
    Xw = Xs * build_rf_knn(df, Xs)[1]  # reuse weights

    # Binary cols only for Jaccard
    bool_cols = [c for c in FEATURE_COLS
                 if df[c].dtype in (bool, np.int64) and c != "noise_encoded"
                 and c not in ("comfort_score","connectivity_score",
                               "group_score","outdoor_score","amenity_score")]
    bool_idx = [FEATURE_COLS.index(c) for c in bool_cols if c in FEATURE_COLS]

    results = []
    for i in range(len(df)):
        true_cat = df.iloc[i]["category"]

        # Plain KNN
        dists, idxs = plain_knn.kneighbors(Xs[i].reshape(1, -1))
        neighbors_plain = [j for j in idxs[0] if j != i][:3]

        # RF-KNN
        dists_rf, idxs_rf = rf_knn.kneighbors(Xw[i].reshape(1, -1))
        neighbors_rf = [j for j in idxs_rf[0] if j != i][:3]

        # SVD
        sim_row = sim_svd[i].copy()
        sim_row[i] = -1
        neighbors_svd = np.argsort(sim_row)[::-1][:3].tolist()

        # Jaccard
        jac_sims = []
        for j in range(len(df)):
            if j == i:
                jac_sims.append(-1.0)
            else:
                jac_sims.append(weighted_jaccard(Xs[i, bool_idx], Xs[j, bool_idx]))
        neighbors_jac = np.argsort(jac_sims)[::-1][:3].tolist()

        for model_name, neighbors in [
            ("Plain KNN",   neighbors_plain),
            ("RF-KNN",      neighbors_rf),
            ("SVD",         neighbors_svd),
            ("Jaccard",     neighbors_jac),
        ]:
            cats = [df.iloc[j]["category"] for j in neighbors]
            p3 = sum(c == true_cat for c in cats) / 3.0
            # MRR: reciprocal rank of first same-category hit
            rr = 0.0
            for rank, c in enumerate(cats, 1):
                if c == true_cat:
                    rr = 1.0 / rank
                    break
            results.append({"spot": df.iloc[i]["name"], "category": true_cat,
                             "model": model_name, "precision_at_3": p3, "mrr": rr})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save(fig, filename):
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> plots/{filename}")


def plot_rf_importance(rf: RandomForestClassifier):
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    imp.index = [FEATURE_LABELS.get(c, c) for c in imp.index]
    imp = imp.sort_values()

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(imp)))
    ax.barh(imp.index, imp.values, color=colors, edgecolor="white")
    ax.axvline(1 / len(imp), color="#888", linestyle="--", linewidth=0.8,
               label=f"Uniform baseline ({1/len(imp):.3f})")
    ax.set_title("Random Forest Feature Importances\n(used as KNN dimension weights)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "11_rf_feature_importance.png")


def plot_model_comparison(eval_df: pd.DataFrame):
    summary = eval_df.groupby("model")[["precision_at_3", "mrr"]].mean().round(3)
    summary = summary.sort_values("precision_at_3", ascending=False)
    summary.to_csv(TABLE_DIR / "model_scores.csv")
    print(f"  saved -> tables/model_scores.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    model_colors = {
        "RF-KNN":     "#2D7D46",
        "Plain KNN":  "#2B6CB0",
        "SVD":        "#C05621",
        "Jaccard":    "#6B46C1",
    }

    for ax, metric, label in zip(
        axes,
        ["precision_at_3", "mrr"],
        ["Precision @ 3 (fraction of top-3 in same category)",
         "Mean Reciprocal Rank (how early same-category appears)"]
    ):
        models = summary.index.tolist()
        vals   = summary[metric].values
        colors = [model_colors.get(m, "#aaa") for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.suptitle("Model Comparison — Study Spot Retrieval Quality", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "12_model_comparison.png")


def plot_pca_spot_map(df: pd.DataFrame, Xs: np.ndarray, weights: np.ndarray):
    """
    2-D PCA of the RF-weighted feature space. Each point is a spot.
    Shows how the models 'see' the spot landscape.
    """
    from sklearn.decomposition import PCA
    Xw = Xs * weights
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xw)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))
    for cat, color in CATEGORY_COLORS.items():
        mask = df["category"] == cat
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=cat.replace("_", " ").title(),
                   s=90, edgecolors="white", linewidths=0.6, zorder=3)
    for i, row in df.iterrows():
        ax.annotate(row["name"].split(" (")[0][:20],
                    (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=5.5, color="#444")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=10)
    ax.set_title("RF-Weighted Feature Space (PCA 2D Projection)\n"
                 "Spots close together will be recommended together",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Category", fontsize=8, title_fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "13_pca_spot_map.png")


def plot_precision_by_category(eval_df: pd.DataFrame):
    pivot = eval_df.pivot_table(
        index="category", columns="model", values="precision_at_3", aggfunc="mean"
    ).round(3)
    model_order = ["RF-KNN", "Plain KNN", "SVD", "Jaccard"]
    pivot = pivot[[m for m in model_order if m in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, edgecolor="white",
               color=["#2D7D46", "#2B6CB0", "#C05621", "#6B46C1"],
               width=0.7)
    ax.set_title("Precision @ 3 by Category and Model",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision @ 3", fontsize=10)
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in pivot.index], rotation=0, fontsize=9
    )
    ax.set_ylim(0, 1.2)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "12b_precision_by_category.png")


def plot_recommendation_heatmap(demo_results: pd.DataFrame):
    """
    For each demo query, show similarity scores across all 30 spots.
    """
    queries = demo_results["query"].unique()
    spots   = demo_results["spot"].unique()

    pivot = demo_results.pivot_table(
        index="query", columns="spot", values="similarity", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(max(16, len(spots) * 0.55), len(queries) + 1))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="#eee",
                annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"shrink": 0.5, "label": "Cosine Similarity"})
    ax.set_title("Recommendation Similarity Scores by Query and Spot",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=6.5)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    save(fig, "14_recommendation_heatmap.png")


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

def build_query_vector(user_prefs: dict) -> np.ndarray:
    """
    Convert a user preference dict into a feature vector matching FEATURE_COLS.

    user_prefs keys (all optional, defaults to neutral):
      wants_wifi        (0/1)   — needs wifi
      wants_outlets     (0/1)   — needs power outlet
      wants_coffee      (0/1)   — wants coffee available
      wants_food        (0/1)   — wants food available
      wants_quiet       (0/1)   — needs quiet environment
      wants_outdoor     (0/1)   — prefers being outside
      wants_group       (0/1)   — group study session
      wants_comfortable (0/1)   — comfortable seating matters
      weather_ok        (0/1)   — ok with weather-dependent spots (default 1)
      noise_tolerance   (0–4)   — 0=need silence, 4=noise fine (default 2)
      connectivity_need (0–3)   — 0=don't care, 3=critical (default 1)
    """
    p = user_prefs

    # Build vector in FEATURE_COLS order
    v = {
        "has_wifi":                float(p.get("wants_wifi", 1)),
        "has_outlets":             float(p.get("wants_outlets", 0)),
        "has_coffee":              float(p.get("wants_coffee", 0)),
        "has_food":                float(p.get("wants_food", 0)),
        "feat_quiet":              float(p.get("wants_quiet", 0)),
        "feat_outdoor":            float(p.get("wants_outdoor", 0)),
        "feat_comfortable_seating":float(p.get("wants_comfortable", 0)),
        "feat_shaded":             float(p.get("wants_outdoor", 0)),   # shaded correlates with outdoor pref
        "feat_benches":            float(p.get("wants_outdoor", 0)),
        "feat_tables":             float(p.get("wants_group", 0)),
        "feat_group_friendly":     float(p.get("wants_group", 0)),
        "feat_printing_nearby":    0.0,
        "feat_natural_light":      float(p.get("wants_outdoor", 0) * 0.5),
        "weather_dependent":       float(p.get("weather_ok", 1)),
        "is_indoor":               float(1 - p.get("wants_outdoor", 0)),
        "is_outdoor":              float(p.get("wants_outdoor", 0)),
        "comfort_score":           float(p.get("wants_comfortable", 0) * 2
                                        + p.get("wants_quiet", 0)),
        "connectivity_score":      float(p.get("connectivity_need", 1)),
        "group_score":             float(p.get("wants_group", 0) * 2),
        "outdoor_score":           float(p.get("wants_outdoor", 0) * 3),
        "amenity_score":           float(p.get("wants_wifi", 1)
                                        + p.get("wants_outlets", 0)
                                        + p.get("wants_coffee", 0)
                                        + p.get("wants_food", 0)),
        "noise_encoded":           float(p.get("noise_tolerance", 2)),
    }
    return np.array([v[c] for c in FEATURE_COLS])


class StudySpotRecommender:
    """
    Content-based recommender using RF-weighted cosine KNN.
    After fit(), call recommend(user_prefs) to get 3 spots.
    """

    def __init__(self):
        self.df      = None
        self.Xs      = None
        self.Xw      = None
        self.scaler  = None
        self.weights = None
        self.rf      = None

    def fit(self, df: pd.DataFrame, Xs: np.ndarray, scaler: MinMaxScaler):
        self.df     = df.reset_index(drop=True)
        self.Xs     = Xs
        self.scaler = scaler
        # Train RF on category labels to get feature weights
        self.rf      = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
        self.rf.fit(Xs, df["category"])
        self.weights = self.rf.feature_importances_
        self.Xw      = Xs * self.weights
        return self

    def recommend(self, user_prefs: dict, n: int = 3,
                  exclude_category: str = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        user_prefs      : dict of user preference values (see build_query_vector)
        n               : number of recommendations (default 3)
        exclude_category: optionally exclude a category from results

        Returns
        -------
        DataFrame with columns: rank, name, category, similarity, [feature cols]
        """
        qv  = build_query_vector(user_prefs)
        qvs = self.scaler.transform(qv.reshape(1, -1))
        qvw = qvs * self.weights

        sims = cosine_similarity(qvw, self.Xw)[0]

        results = self.df.copy()
        results["similarity"] = sims

        if exclude_category:
            results = results[results["category"] != exclude_category]

        results = results.sort_values("similarity", ascending=False).head(n)
        results.insert(0, "rank", range(1, len(results) + 1))

        return results[["rank", "name", "category", "similarity"] + FEATURE_COLS]

    def explain(self, spot_name: str, user_prefs: dict) -> pd.DataFrame:
        """
        Per-feature contribution breakdown for a given spot recommendation.
        Shows which features drove the match.
        """
        qv  = build_query_vector(user_prefs)
        qvs = self.scaler.transform(qv.reshape(1, -1))[0]

        idx  = self.df[self.df["name"] == spot_name].index[0]
        spot = self.Xs[idx]

        rows = []
        for i, feat in enumerate(FEATURE_COLS):
            q_val   = float(qvs[i])
            s_val   = float(spot[i])
            weight  = float(self.weights[i])
            contrib = float(min(q_val, s_val) * weight)   # weighted overlap
            rows.append({
                "feature":      FEATURE_LABELS.get(feat, feat),
                "user_value":   round(q_val, 3),
                "spot_value":   round(s_val, 3),
                "rf_weight":    round(weight, 4),
                "contribution": round(contrib, 4),
            })
        return pd.DataFrame(rows).sort_values("contribution", ascending=False)


# ---------------------------------------------------------------------------
# Demo queries
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    {
        "name": "Quiet solo study, need wifi + outlets, indoors",
        "prefs": {
            "wants_wifi": 1, "wants_outlets": 1, "wants_coffee": 0,
            "wants_food": 0, "wants_quiet": 1, "wants_outdoor": 0,
            "wants_group": 0, "wants_comfortable": 1,
            "weather_ok": 0, "noise_tolerance": 0, "connectivity_need": 3,
        },
    },
    {
        "name": "Group study, need food + coffee, noise is fine",
        "prefs": {
            "wants_wifi": 1, "wants_outlets": 0, "wants_coffee": 1,
            "wants_food": 1, "wants_quiet": 0, "wants_outdoor": 0,
            "wants_group": 1, "wants_comfortable": 0,
            "weather_ok": 1, "noise_tolerance": 4, "connectivity_need": 1,
        },
    },
    {
        "name": "Outdoor, shaded, reading — no tech needed",
        "prefs": {
            "wants_wifi": 0, "wants_outlets": 0, "wants_coffee": 0,
            "wants_food": 0, "wants_quiet": 1, "wants_outdoor": 1,
            "wants_group": 0, "wants_comfortable": 0,
            "weather_ok": 1, "noise_tolerance": 1, "connectivity_need": 0,
        },
    },
    {
        "name": "Quick coffee + laptop, moderate noise ok",
        "prefs": {
            "wants_wifi": 1, "wants_outlets": 1, "wants_coffee": 1,
            "wants_food": 0, "wants_quiet": 0, "wants_outdoor": 0,
            "wants_group": 0, "wants_comfortable": 0,
            "weather_ok": 0, "noise_tolerance": 3, "connectivity_need": 2,
        },
    },
    {
        "name": "Very quiet, comfortable, no food needed",
        "prefs": {
            "wants_wifi": 0, "wants_outlets": 0, "wants_coffee": 0,
            "wants_food": 0, "wants_quiet": 1, "wants_outdoor": 0,
            "wants_group": 0, "wants_comfortable": 1,
            "weather_ok": 0, "noise_tolerance": 0, "connectivity_need": 0,
        },
    },
]


def run_demo(recommender: StudySpotRecommender) -> pd.DataFrame:
    all_rows = []
    print("\n" + "="*62)
    print("  DEMO RECOMMENDATIONS")
    print("="*62)

    for query in DEMO_QUERIES:
        name  = query["name"]
        prefs = query["prefs"]
        recs  = recommender.recommend(prefs, n=3)

        print(f"\n  Query: {name}")
        print(f"  {'Rank':<5} {'Spot':<42} {'Category':<14} {'Sim':>5}")
        print(f"  {'-'*68}")
        for _, row in recs.iterrows():
            print(f"  {int(row['rank']):<5} {row['name'][:40]:<42} "
                  f"{row['category']:<14} {row['similarity']:.3f}")

        # Print explanation for top rec
        top_name = recs.iloc[0]["name"]
        explain  = recommender.explain(top_name, prefs)
        print(f"\n  Top match explanation: {top_name}")
        print(f"  {'Feature':<25} {'User':>6} {'Spot':>6} {'Weight':>7} {'Contrib':>8}")
        for _, r in explain.head(5).iterrows():
            print(f"  {r['feature'][:24]:<25} {r['user_value']:>6.2f} "
                  f"{r['spot_value']:>6.2f} {r['rf_weight']:>7.4f} {r['contribution']:>8.4f}")

        # Collect for heatmap
        sims_all = cosine_similarity(
            (recommender.scaler.transform(
                build_query_vector(prefs).reshape(1,-1)) * recommender.weights),
            recommender.Xw
        )[0]
        for i, spot_name in enumerate(recommender.df["name"]):
            all_rows.append({
                "query": name[:35],
                "spot":  spot_name[:22],
                "similarity": round(float(sims_all[i]), 3),
            })

    print("\n" + "="*62 + "\n")
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nLoading data ...")
    df, Xs, scaler = load_and_prepare()
    print(f"  {len(df)} spots, {len(FEATURE_COLS)} features after engineering")

    # --- RF for importance + explanation ---
    print("\nTraining Random Forest (feature weighting) ...")
    rf_knn, weights, rf = build_rf_knn(df, Xs)
    plot_rf_importance(rf)

    # --- LOO cross-val of RF as sanity check ---
    print("\nRandom Forest LOO cross-validation (category prediction) ...")
    loo_scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        Xs, df["category"], cv=LeaveOneOut(), scoring="accuracy"
    )
    print(f"  LOO accuracy: {loo_scores.mean():.3f}  (n=30, 4 classes, "
          f"chance={1/4:.2f})")

    # --- Evaluate all models ---
    print("\nEvaluating all retrieval models ...")
    eval_df = evaluate_models(df, Xs)
    summary = eval_df.groupby("model")[["precision_at_3","mrr"]].mean().round(3)
    print("\n  Model comparison (mean across 30 spots):")
    print(f"  {'Model':<12} {'Precision@3':>12} {'MRR':>8}")
    print(f"  {'-'*34}")
    for model, row in summary.sort_values("precision_at_3", ascending=False).iterrows():
        print(f"  {model:<12} {row['precision_at_3']:>12.3f} {row['mrr']:>8.3f}")

    plot_model_comparison(eval_df)
    plot_precision_by_category(eval_df)
    plot_pca_spot_map(df, Xs, weights)

    # --- Build final recommender (RF-KNN) ---
    print("\nBuilding recommender ...")
    rec = StudySpotRecommender().fit(df, Xs, scaler)

    # --- Run demo queries ---
    demo_df = run_demo(rec)
    plot_recommendation_heatmap(demo_df)

    # Save demo results
    recs_rows = []
    for query in DEMO_QUERIES:
        recs = rec.recommend(query["prefs"], n=3)
        for _, row in recs.iterrows():
            recs_rows.append({
                "query":      query["name"],
                "rank":       row["rank"],
                "name":       row["name"],
                "category":   row["category"],
                "similarity": round(row["similarity"], 4),
            })
    recs_df = pd.DataFrame(recs_rows)
    recs_df.to_csv(TABLE_DIR / "recommendations_demo.csv", index=False)
    print(f"  saved -> tables/recommendations_demo.csv")

    print("\nDone. All outputs written to plots/ and tables/\n")


if __name__ == "__main__":
    main()