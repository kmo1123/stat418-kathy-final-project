"""
ucla_study_spots_eda.py
------------------------
Exploratory data analysis of the UCLA study spots dataset.

Sections:
  1. Load & normalize        — flatten JSON into a unified DataFrame
  2. Feature engineering     — OHE, ordinal encoding, derived features
  3. Summary tables          — per-category counts, feature coverage
  4. Univariate plots         — category distribution, noise level, feature freq
  5. Bivariate / cross-tabs  — noise by category, amenity co-occurrence
  6. Correlation matrix       — numeric + encoded features
  7. Geographic plot          — lat/lng scatter colored by category
  8. Timing analysis          — best/avoid time slot frequency heatmap

Outputs:
  plots/01_category_dist.png
  plots/02_noise_level.png
  plots/03_feature_frequency.png
  plots/04_noise_by_category.png
  plots/05_amenity_cooccurrence.png
  plots/06_correlation_matrix.png
  plots/07_geographic.png
  plots/08_timing_heatmap.png
  plots/09_feature_coverage_by_category.png
  tables/feature_matrix.csv
  tables/summary_stats.csv
  tables/timing_slots.csv

Usage:
  python eda.py
"""

from pathlib import Path
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "raw" / "ucla" / "all_spots.json"
PLOT_DIR  = BASE_DIR / "plots"
TABLE_DIR = BASE_DIR / "tables"
PLOT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Palette — one color per category, used consistently across all plots
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "library":      "#2B6CB0",
    "study_lounge": "#2D7D46",
    "cafe":         "#C05621",
    "outdoor":      "#6B46C1",
}

NOISE_ORDER = ["silent", "very_quiet", "quiet", "low_moderate", "moderate", "high"]

# ---------------------------------------------------------------------------
# 1. Load & normalize
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    spots = json.loads(path.read_text(encoding="utf-8"))

    # All known feature tags across dataset
    ALL_FEATURES = [
        "wifi", "outlets", "coffee", "food", "quiet", "outdoor",
        "comfortable_seating", "shaded", "benches", "tables",
        "group_friendly", "printing_nearby", "outdoor_patio",
        "outdoor_seating", "natural_light", "flexible_seating",
        "library_adjacent", "private_carrels", "group_rooms",
        "writable_walls", "meal_vouchers", "picnic_friendly",
        "shaded_tables", "coffee_adjacent", "low_traffic",
        "central_location", "sheltered_atrium", "nature",
        "walking_paths", "free_admission",
    ]

    rows = []
    for s in spots:
        features = s.get("features") or []
        row = {
            # identity
            "name":              s["name"],
            "category":          s["category"],
            "source":            s.get("source", ""),
            # location
            "lat":               s.get("lat"),
            "lng":               s.get("lng"),
            "area":              s.get("area") or s.get("building", ""),
            # noise (keep raw string — encoded separately)
            "noise_level":       s.get("noise_level"),
            # boolean amenities — check both top-level fields (cafes) and feature tags
            "has_coffee":        bool(s.get("coffee")) or "coffee" in (s.get("features") or []),
            "has_food":          bool(s.get("food"))   or "food"   in (s.get("features") or []),
            "has_wifi":          bool(s.get("wifi"))   or "wifi"   in (s.get("features") or []),
            "has_outlets":       bool(s.get("outlets")) or "outlets" in (s.get("features") or []),
            # boolean flags
            "food_drink":        bool(s.get("food_drink")),
            "weather_dependent": bool(s.get("weather_dependent")),
            "is_indoor":         s.get("indoor_outdoor") in ("indoor", "both"),
            "is_outdoor":        s.get("indoor_outdoor") in ("outdoor", "both")
                                 or "outdoor" in features,
            # timing
            "n_best_times":      len(s.get("best_times") or []),
            "n_avoid_times":     len(s.get("avoid_times") or []),
            "has_avoid_times":   len(s.get("avoid_times") or []) > 0,
            # feature count
            "n_features":        len(features),
            # raw timing lists (for slot analysis)
            "_best_times":       s.get("best_times") or [],
            "_avoid_times":      s.get("avoid_times") or [],
        }
        # One-hot encode every feature tag
        for feat in ALL_FEATURES:
            row[f"feat_{feat}"] = int(feat in features)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, ALL_FEATURES


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered columns on top of the raw normalized frame.

    OHE columns (already done in load_data via feat_* prefix):
      - One column per feature tag, value 0/1.

    Ordinal encoding:
      - noise_encoded: maps noise_level string → integer (0=silent … 5=high)
        Libraries have no noise_level — filled with category median after encoding.

    Derived features:
      - amenity_score: sum of the four core boolean amenities
        (wifi + outlets + coffee + food) — 0–4 scale.
      - is_food_capable: True if the spot has food OR coffee OR food_drink.
      - comfort_score: sum of comfort-related feature tags
        (comfortable_seating, quiet, natural_light, shaded, private_carrels).
      - connectivity_score: sum of connectivity tags
        (wifi, outlets, printing_nearby).
      - group_score: sum of group-oriented tags
        (group_friendly, group_rooms, flexible_seating, writable_walls, tables).
      - outdoor_score: sum of outdoor-quality tags
        (outdoor, shaded, shaded_tables, benches, tables, picnic_friendly,
         walking_paths, nature).
      - category_encoded: integer label for category (for correlation matrix).
    """

    # --- Ordinal: noise level ---
    noise_map = {
        "silent":       0,
        "very_quiet":   1,
        "quiet":        2,
        "low_moderate": 3,
        "moderate":     4,
        "high":         5,
    }
    df["noise_encoded"] = df["noise_level"].map(noise_map)
    # Libraries have no noise_level — impute with category median
    cat_medians = df.groupby("category")["noise_encoded"].median()
    df["noise_encoded"] = df.apply(
        lambda r: cat_medians[r["category"]] if pd.isna(r["noise_encoded"]) else r["noise_encoded"],
        axis=1,
    )

    # --- OHE: category ---
    cat_order = ["library", "study_lounge", "cafe", "outdoor"]
    df["category_encoded"] = df["category"].map({c: i for i, c in enumerate(cat_order)})

    # --- Amenity score (0–4): wifi + outlets + coffee + food ---
    df["amenity_score"] = (
        df["has_wifi"].astype(int)
        + df["has_outlets"].astype(int)
        + df["has_coffee"].astype(int)
        + df["has_food"].astype(int)
    )

    # --- Food capable ---
    df["is_food_capable"] = (
        df["has_coffee"] | df["has_food"] | df["food_drink"]
    )

    # --- Composite scores from feature tags ---
    comfort_tags = ["feat_comfortable_seating", "feat_quiet", "feat_natural_light",
                    "feat_shaded", "feat_private_carrels"]
    connectivity_tags = ["feat_wifi", "feat_outlets", "feat_printing_nearby"]
    group_tags = ["feat_group_friendly", "feat_group_rooms", "feat_flexible_seating",
                  "feat_writable_walls", "feat_tables"]
    outdoor_tags = ["feat_outdoor", "feat_shaded", "feat_shaded_tables", "feat_benches",
                    "feat_tables", "feat_picnic_friendly", "feat_walking_paths", "feat_nature"]

    df["comfort_score"]      = df[[t for t in comfort_tags      if t in df.columns]].sum(axis=1)
    df["connectivity_score"] = df[[t for t in connectivity_tags  if t in df.columns]].sum(axis=1)
    df["group_score"]        = df[[t for t in group_tags         if t in df.columns]].sum(axis=1)
    df["outdoor_score"]      = df[[t for t in outdoor_tags       if t in df.columns]].sum(axis=1)

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, filename: str):
    path = PLOT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> plots/{filename}")


def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# 3. Summary tables
# ---------------------------------------------------------------------------

def make_summary_tables(df: pd.DataFrame, all_features: list):
    # Per-category summary
    summary = df.groupby("category").agg(
        count=("name", "count"),
        pct_wifi=("has_wifi", "mean"),
        pct_outlets=("has_outlets", "mean"),
        pct_food_capable=("is_food_capable", "mean"),
        pct_weather_dep=("weather_dependent", "mean"),
        mean_n_features=("n_features", "mean"),
        mean_noise_encoded=("noise_encoded", "mean"),
        mean_comfort_score=("comfort_score", "mean"),
        mean_connectivity_score=("connectivity_score", "mean"),
        mean_group_score=("group_score", "mean"),
    ).round(2)
    summary.to_csv(TABLE_DIR / "summary_stats.csv")
    print(f"  saved -> tables/summary_stats.csv")

    # Feature presence matrix (spots × features)
    feat_cols = [f"feat_{f}" for f in all_features if f"feat_{f}" in df.columns]
    feat_matrix = df[["name", "category"] + feat_cols].copy()
    feat_matrix.columns = ["name", "category"] + all_features[: len(feat_cols)]
    feat_matrix.to_csv(TABLE_DIR / "feature_matrix.csv", index=False)
    print(f"  saved -> tables/feature_matrix.csv")

    return summary


# ---------------------------------------------------------------------------
# 4. Univariate plots
# ---------------------------------------------------------------------------

def plot_category_distribution(df: pd.DataFrame):
    counts = df["category"].value_counts().reindex(
        ["library", "study_lounge", "cafe", "outdoor"]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[CATEGORY_COLORS[c] for c in counts.index],
        edgecolor="white", linewidth=0.8,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    style_ax(ax, "Spots by Category", ylabel="Count")
    ax.set_ylim(0, counts.max() + 2)
    ax.set_xticklabels([c.replace("_", " ").title() for c in counts.index])
    fig.tight_layout()
    save(fig, "01_category_dist.png")


def plot_noise_distribution(df: pd.DataFrame):
    # Only spots with a noise_level value
    noise_df = df.dropna(subset=["noise_level"])
    order = [n for n in NOISE_ORDER if n in noise_df["noise_level"].values]
    counts = noise_df["noise_level"].value_counts().reindex(order).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("Blues_r", len(order))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    style_ax(ax, "Noise Level Distribution (non-library spots)",
             xlabel="Noise Level", ylabel="Count")
    ax.set_xticklabels([n.replace("_", " ").title() for n in order])
    fig.tight_layout()
    save(fig, "02_noise_level.png")


def plot_feature_frequency(df: pd.DataFrame, all_features: list):
    feat_cols = [f"feat_{f}" for f in all_features if f"feat_{f}" in df.columns]
    freq = df[feat_cols].sum().sort_values(ascending=True)
    freq.index = [i.replace("feat_", "").replace("_", " ") for i in freq.index]
    freq = freq[freq > 0]  # drop zero-frequency tags

    fig, ax = plt.subplots(figsize=(8, max(5, len(freq) * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(freq)))
    ax.barh(freq.index, freq.values, color=colors, edgecolor="white")
    for i, val in enumerate(freq.values):
        ax.text(val + 0.05, i, str(int(val)), va="center", fontsize=8)
    style_ax(ax, "Feature Tag Frequency (across all 30 spots)",
             xlabel="Number of Spots")
    ax.set_xlim(0, freq.max() + 2)
    fig.tight_layout()
    save(fig, "03_feature_frequency.png")


# ---------------------------------------------------------------------------
# 5. Bivariate / cross-tab plots
# ---------------------------------------------------------------------------

def plot_noise_by_category(df: pd.DataFrame):
    # Stacked bar: category × noise level
    noise_df = df.dropna(subset=["noise_level"]).copy()
    noise_df["noise_level"] = pd.Categorical(noise_df["noise_level"], categories=NOISE_ORDER)
    cross = (
        noise_df.groupby(["category", "noise_level"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    # Keep only levels that appear
    cross = cross[[c for c in NOISE_ORDER if c in cross.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    noise_palette = sns.color_palette("Blues", len(cross.columns))
    cross.plot(kind="bar", ax=ax, color=noise_palette, edgecolor="white",
               stacked=True, width=0.6)
    style_ax(ax, "Noise Level by Category", ylabel="Number of Spots")
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in cross.index], rotation=0
    )
    handles = [
        mpatches.Patch(color=noise_palette[i], label=c.replace("_", " ").title())
        for i, c in enumerate(cross.columns)
    ]
    ax.legend(handles=handles, title="Noise Level", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "04_noise_by_category.png")


def plot_amenity_cooccurrence(df: pd.DataFrame):
    """
    Heatmap showing how often pairs of feature tags appear together.
    Only includes tags that appear in at least 3 spots.
    """
    feat_cols = [c for c in df.columns if c.startswith("feat_") and df[c].sum() >= 3]
    labels    = [c.replace("feat_", "").replace("_", " ") for c in feat_cols]
    sub = df[feat_cols].astype(int)
    cooc = sub.T.dot(sub)
    cooc.index   = labels
    cooc.columns = labels

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.eye(len(cooc), dtype=bool)  # hide diagonal (self-occurrence)
    sns.heatmap(
        cooc, ax=ax, mask=mask, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.5, linecolor="#eee", annot_kws={"size": 8},
        cbar_kws={"shrink": 0.7},
    )
    style_ax(ax, "Feature Tag Co-occurrence (spots with tag A and tag B)")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save(fig, "05_amenity_cooccurrence.png")


def plot_feature_coverage_by_category(df: pd.DataFrame, all_features: list):
    """
    Grouped bar: for each category, what % of spots have each core feature.
    Shows the 8 most common features only for readability.
    """
    top_feats = ["feat_wifi", "feat_outlets", "feat_coffee", "feat_food",
                 "feat_quiet", "feat_outdoor", "feat_comfortable_seating", "feat_shaded"]
    top_feats = [f for f in top_feats if f in df.columns]
    labels = [f.replace("feat_", "").replace("_", " ").title() for f in top_feats]

    cats = ["library", "study_lounge", "cafe", "outdoor"]
    data = {}
    for cat in cats:
        sub = df[df["category"] == cat]
        data[cat] = [sub[f].mean() * 100 for f in top_feats]

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cat in enumerate(cats):
        ax.bar(x + i * width, data[cat], width,
               label=cat.replace("_", " ").title(),
               color=CATEGORY_COLORS[cat], edgecolor="white")

    style_ax(ax, "Core Feature Coverage by Category (%)",
             ylabel="% of Spots with Feature")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    save(fig, "09_feature_coverage_by_category.png")


# ---------------------------------------------------------------------------
# 6. Correlation matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame):
    """
    Pearson correlation across all numeric and engineered features.
    Includes: noise_encoded, category_encoded, all amenity/composite scores,
    boolean flags, and n_features / timing counts.
    """
    num_cols = [
        "category_encoded", "noise_encoded",
        "has_wifi", "has_outlets", "has_coffee", "has_food",
        "weather_dependent", "is_indoor", "is_outdoor", "is_food_capable",
        "amenity_score", "comfort_score", "connectivity_score",
        "group_score", "outdoor_score",
        "n_features", "n_best_times", "n_avoid_times",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].astype(float).corr()

    pretty_labels = {
        "category_encoded":    "Category",
        "noise_encoded":       "Noise Level",
        "has_wifi":            "WiFi",
        "has_outlets":         "Outlets",
        "has_coffee":          "Coffee",
        "has_food":            "Food",
        "weather_dependent":   "Weather Dep.",
        "is_indoor":           "Indoor",
        "is_outdoor":          "Outdoor",
        "is_food_capable":     "Food Capable",
        "amenity_score":       "Amenity Score",
        "comfort_score":       "Comfort Score",
        "connectivity_score":  "Connectivity",
        "group_score":         "Group Score",
        "outdoor_score":       "Outdoor Score",
        "n_features":          "# Features",
        "n_best_times":        "# Best Times",
        "n_avoid_times":       "# Avoid Times",
    }
    corr.index   = [pretty_labels.get(c, c) for c in corr.index]
    corr.columns = [pretty_labels.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle
    sns.heatmap(
        corr, ax=ax, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="#ddd",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )
    style_ax(ax, "Correlation Matrix — Numeric & Engineered Features")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    save(fig, "06_correlation_matrix.png")


# ---------------------------------------------------------------------------
# 7. Geographic scatter
# ---------------------------------------------------------------------------

def plot_geographic(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 7))

    for cat, color in CATEGORY_COLORS.items():
        sub = df[df["category"] == cat]
        ax.scatter(
            sub["lng"], sub["lat"],
            c=color, label=cat.replace("_", " ").title(),
            s=80, edgecolors="white", linewidths=0.6, zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["name"].split(" (")[0][:22],  # truncate long names
                (row["lng"], row["lat"]),
                textcoords="offset points", xytext=(5, 3),
                fontsize=5.5, color="#333",
            )

    style_ax(ax, "UCLA Study Spots — Geographic Distribution",
             xlabel="Longitude", ylabel="Latitude")
    ax.legend(title="Category", fontsize=8, title_fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    save(fig, "07_geographic.png")


# ---------------------------------------------------------------------------
# 8. Timing slot heatmap
# ---------------------------------------------------------------------------

# Map free-text timing strings to canonical time slots
TIME_SLOT_PATTERNS = [
    ("Early Morning (before 9am)",  r"before 9|before 9:30|early morning|7am"),
    ("Morning (9am–11am)",          r"morning|before 10|before 11|9am|10am"),
    ("Midday (11am–2pm)",           r"11am|noon|12pm|lunch|midday|11:30"),
    ("Afternoon (2pm–5pm)",         r"after 2|after 3|afternoon|mid.afternoon|2pm|3pm|4pm"),
    ("Evening (5pm–8pm)",           r"after 7|evening|7pm|after 4|after 5|6pm|5pm"),
    ("Night (after 8pm)",           r"after 8|overnight|night powell|late night"),
    ("Weekdays",                    r"weekday"),
    ("Weekends",                    r"weekend|friday afternoon|saturday|sunday"),
    ("Finals / Midterms",           r"finals|midterm|exam"),
]

def _map_slots(text_list: list) -> list:
    slots = []
    combined = " ".join(text_list).lower()
    for slot_name, pattern in TIME_SLOT_PATTERNS:
        if re.search(pattern, combined):
            slots.append(slot_name)
    return slots


def plot_timing_heatmap(df: pd.DataFrame):
    slot_names = [s for s, _ in TIME_SLOT_PATTERNS]
    cats       = ["library", "study_lounge", "cafe", "outdoor"]

    best_matrix  = pd.DataFrame(0, index=cats, columns=slot_names)
    avoid_matrix = pd.DataFrame(0, index=cats, columns=slot_names)

    for _, row in df.iterrows():
        cat = row["category"]
        for slot in _map_slots(row["_best_times"]):
            best_matrix.loc[cat, slot] += 1
        for slot in _map_slots(row["_avoid_times"]):
            avoid_matrix.loc[cat, slot] += 1

    # Save timing slot table
    timing_export = pd.concat(
        [best_matrix.add_suffix("_best"), avoid_matrix.add_suffix("_avoid")], axis=1
    )
    timing_export.to_csv(TABLE_DIR / "timing_slots.csv")
    print(f"  saved -> tables/timing_slots.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    def _draw_hm(mat, ax, title, cmap):
        sns.heatmap(
            mat, ax=ax, annot=True, fmt="d", cmap=cmap,
            linewidths=0.5, linecolor="#eee",
            annot_kws={"size": 9},
            cbar_kws={"shrink": 0.7},
            vmin=0,
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(
            [c.replace("_", " ").title() for c in mat.index], rotation=0, fontsize=9
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    _draw_hm(best_matrix,  ax1, "Best Times to Visit (spot count per slot)", "Greens")
    _draw_hm(avoid_matrix, ax2, "Times to Avoid (spot count per slot)",       "Reds")
    fig.suptitle("Timing Patterns by Category", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "08_timing_heatmap.png")


# ---------------------------------------------------------------------------
# 9. Composite score comparison
# ---------------------------------------------------------------------------

def plot_composite_scores(df: pd.DataFrame):
    """
    Radar-style grouped bar showing mean composite scores per category.
    """
    scores = ["comfort_score", "connectivity_score", "group_score",
              "outdoor_score", "amenity_score"]
    labels = ["Comfort", "Connectivity", "Group\nFriendly", "Outdoor\nQuality", "Amenities"]
    cats   = ["library", "study_lounge", "cafe", "outdoor"]

    means = df.groupby("category")[scores].mean()
    # Normalize each score to 0–1 range for fair comparison
    for col in scores:
        col_max = means[col].max()
        if col_max > 0:
            means[col] = means[col] / col_max

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cat in enumerate(cats):
        vals = [means.loc[cat, s] if cat in means.index else 0 for s in scores]
        ax.bar(x + i * width, vals, width,
               label=cat.replace("_", " ").title(),
               color=CATEGORY_COLORS[cat], edgecolor="white")

    style_ax(ax, "Normalized Composite Scores by Category",
             ylabel="Score (normalized 0–1)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.2)
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "10_composite_scores.png")


# ---------------------------------------------------------------------------
# Print summary to stdout
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, summary: pd.DataFrame):
    print("\n" + "="*60)
    print("  DATASET OVERVIEW")
    print("="*60)
    print(f"  Total spots:      {len(df)}")
    print(f"  Categories:       {df['category'].nunique()}")
    print(f"  Unique features:  {df[[c for c in df.columns if c.startswith('feat_')]].sum(axis=0).astype(bool).sum()}")
    print(f"  Spots with WiFi:  {df['has_wifi'].sum()}")
    print(f"  Spots w/ outlets: {df['has_outlets'].sum()}")
    print(f"  Outdoor spots:    {df['is_outdoor'].sum()}")
    print()
    print("  PER-CATEGORY SUMMARY")
    print("-"*60)
    print(summary[["count","pct_wifi","pct_outlets","mean_n_features","mean_noise_encoded"]].to_string())
    print()
    print("  ENGINEERED FEATURE STATS")
    print("-"*60)
    eng_cols = ["amenity_score","comfort_score","connectivity_score","group_score","outdoor_score"]
    print(df.groupby("category")[eng_cols].mean().round(2).to_string())
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\nLoading data from {DATA_FILE} ...")
    df, all_features = load_data(DATA_FILE)
    df = engineer_features(df)
    print(f"  {len(df)} spots loaded, {len(df.columns)} columns after engineering\n")

    print("Generating summary tables ...")
    summary = make_summary_tables(df, all_features)

    print("\nGenerating plots ...")
    plot_category_distribution(df)
    plot_noise_distribution(df)
    plot_feature_frequency(df, all_features)
    plot_noise_by_category(df)
    plot_amenity_cooccurrence(df)
    plot_feature_coverage_by_category(df, all_features)
    plot_correlation_matrix(df)
    plot_geographic(df)
    plot_timing_heatmap(df)
    plot_composite_scores(df)

    print_summary(df, summary)

    # Save engineered DataFrame for downstream use
    out = TABLE_DIR / "engineered_df.csv"
    df.drop(columns=["_best_times", "_avoid_times"]).to_csv(out, index=False)
    print(f"  saved -> tables/engineered_df.csv")
    print("Done.\n")


if __name__ == "__main__":
    main()