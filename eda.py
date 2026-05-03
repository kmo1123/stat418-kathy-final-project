"""
ucla_study_spots_eda.py
-----------------------
Exploratory Data Analysis for UCLA on-campus study spots dataset.

Sections:
  1. Load & parse raw data
  2. Key visualisations
  3. Pattern discovery / insights
  4. Data quality audit
  5. Feature engineering (study_score + dist_to_nearest_cafe_m only)

Run:
  pip install pandas matplotlib seaborn numpy
  python eda.py
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import json
import matplotlib
matplotlib.use('Agg')  # No popups — save directly to files

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
UCLA_BLUE   = "#2774AE"
UCLA_GOLD   = "#FFD100"
UCLA_DARK   = "#003B5C"
UCLA_LIGHT  = "#8BB8E8"

sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD & PARSE DATA
# ═══════════════════════════════════════════════════════════════════════════════

INLINE_DATA = [
    # ── Libraries ──────────────────────────────────────────────────────────
    {"name":"Powell Library","category":"library","lat":34.0713,"lng":-118.4417,
     "noise_level":None,"features":["wifi","24_hours","quiet","group_rooms","printing","outlets","computers","reservable"],
     "wifi":True,"outlets":True,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Main undergrad library. Night Powell opens overnight during finals week."},
    {"name":"Young Research Library (YRL)","category":"library","lat":34.0752,"lng":-118.4418,
     "noise_level":None,"features":["wifi","quiet","group_rooms","printing","outlets","computers","3d_printing","reservable","food_nearby"],
     "wifi":True,"outlets":True,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Graduate-focused. Cafe 451 on 1st floor."},
    {"name":"Science & Engineering Library","category":"library","lat":34.0692,"lng":-118.4432,
     "noise_level":None,"features":["wifi","quiet","printing","outlets","computers","reservable"],
     "wifi":True,"outlets":True,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Silent section at far end of Boelter is legendary."},
    {"name":"Biomedical Library","category":"library","lat":34.0663,"lng":-118.4446,
     "noise_level":None,"features":["wifi","24_hours","quiet","group_rooms","outlets","reservable"],
     "wifi":True,"outlets":True,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"24-hour reading room for health/life sciences grad students."},
    {"name":"Music Library","category":"library","lat":34.0704,"lng":-118.4398,
     "noise_level":None,"features":["wifi","quiet"],
     "wifi":True,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Very small and very quiet — hidden gem."},
    {"name":"East Asian Library","category":"library","lat":34.0752,"lng":-118.4415,
     "noise_level":None,"features":["wifi","quiet"],
     "wifi":True,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Tucked inside YRL. Very quiet."},
    {"name":"Management Library","category":"library","lat":34.0730,"lng":-118.4397,
     "noise_level":None,"features":["wifi","quiet","food_nearby"],
     "wifi":True,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Inside Anderson School. Starbucks nearby."},
    {"name":"Arts Library","category":"library","lat":34.0736,"lng":-118.4393,
     "noise_level":None,"features":["wifi","quiet","natural_light"],
     "wifi":True,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Good natural light. Less crowded."},
    # ── Study Lounges ──────────────────────────────────────────────────────
    {"name":"Kerckhoff 3rd Floor Lounge","category":"study_lounge","lat":34.0713,"lng":-118.4440,
     "noise_level":"quiet","features":["wifi","quiet","outlets","comfortable_seating"],
     "wifi":True,"outlets":True,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Most beautiful study spot on campus. Get there early."},
    {"name":"Kerckhoff 2nd Floor Lounge","category":"study_lounge","lat":34.0713,"lng":-118.4440,
     "noise_level":"quiet","features":["wifi","quiet","coffee_nearby"],
     "wifi":True,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Less known than 3rd floor — easier to get a seat."},
    {"name":"Ackerman Union Open Study","category":"study_lounge","lat":34.0707,"lng":-118.4440,
     "noise_level":"moderate","features":["wifi","group_friendly","printing"],
     "wifi":True,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Hit or miss depending on room availability."},
    {"name":"Bruin Reflection Space","category":"study_lounge","lat":34.0707,"lng":-118.4440,
     "noise_level":"silent","features":["wifi","quiet","comfortable_seating"],
     "wifi":True,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Underrated for solo focused work. Very few know about it."},
    {"name":"Transfer Student Center Lounge","category":"study_lounge","lat":34.0707,"lng":-118.4440,
     "noise_level":"moderate","features":["wifi","comfortable_seating"],
     "wifi":True,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Welcoming and rarely crowded."},
    {"name":"Veteran Resource Center","category":"study_lounge","lat":34.0713,"lng":-118.4440,
     "noise_level":"quiet","features":["wifi","comfortable_seating"],
     "wifi":True,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Tight-knit community for veteran students."},
    # ── Cafes ──────────────────────────────────────────────────────────────
    {"name":"Kerckhoff Coffeehouse","category":"cafe","lat":34.0713,"lng":-118.4440,
     "noise_level":"moderate","features":["wifi","outlets","coffee","food","historic_building","outdoor"],
     "wifi":True,"outlets":True,"food_drink":True,"coffee":True,"food":True,
     "weather_dependent":False,"insider_note":"Iconic but fills fast. Outdoor patio less crowded."},
    {"name":"Cafe 451","category":"cafe","lat":34.0752,"lng":-118.4418,
     "noise_level":"moderate","features":["wifi","outlets","coffee","food","quiet_nearby"],
     "wifi":True,"outlets":True,"food_drink":True,"coffee":True,"food":True,
     "weather_dependent":False,"insider_note":"Coffee here then up to quiet YRL stacks."},
    {"name":"Jimmy's Coffee House","category":"cafe","lat":34.0738,"lng":-118.4408,
     "noise_level":"low_moderate","features":["wifi","outlets","coffee","food","outdoor"],
     "wifi":True,"outlets":True,"food_drink":True,"coffee":True,"food":True,
     "weather_dependent":False,"insider_note":"Less hectic than Kerckhoff. Student favourite."},
    {"name":"The Study at Hedrick","category":"cafe","lat":34.0731,"lng":-118.4484,
     "noise_level":"quiet","features":["wifi","outlets","coffee","food","quiet","group_rooms","natural_light"],
     "wifi":True,"outlets":True,"food_drink":True,"coffee":True,"food":True,
     "weather_dependent":False,"insider_note":"Arguably the best study cafe on campus."},
    {"name":"Music Cafe","category":"cafe","lat":34.0706,"lng":-118.4396,
     "noise_level":"low_moderate","features":["wifi","outlets","coffee","food"],
     "wifi":True,"outlets":True,"food_drink":True,"coffee":True,"food":True,
     "weather_dependent":False,"insider_note":"Less known than Kerckhoff but equally charming."},
    {"name":"Bruin Cafe","category":"cafe","lat":34.0728,"lng":-118.4476,
     "noise_level":"moderate","features":["wifi","outlets","coffee","food","outdoor"],
     "wifi":True,"outlets":True,"food_drink":True,"coffee":True,"food":True,
     "weather_dependent":False,"insider_note":"Convenient for Hill residents."},
    {"name":"Anderson School Starbucks","category":"cafe","lat":34.0730,"lng":-118.4397,
     "noise_level":"moderate","features":["wifi","coffee","food","outdoor"],
     "wifi":True,"outlets":False,"food_drink":True,"coffee":True,"food":True,
     "weather_dependent":False,"insider_note":"Quick coffee before Management Library."},
    {"name":"North Campus Student Center","category":"cafe","lat":34.0753,"lng":-118.4404,
     "noise_level":"moderate","features":["wifi","outlets","coffee","food","outdoor","group_friendly"],
     "wifi":True,"outlets":True,"food_drink":True,"coffee":True,"food":True,
     "weather_dependent":False,"insider_note":"Great base for long study days."},
    # ── Outdoor ────────────────────────────────────────────────────────────
    {"name":"Murphy Sculpture Garden","category":"outdoor","lat":34.0754,"lng":-118.4396,
     "noise_level":"quiet","features":["outdoor","peaceful","benches","shaded","picnic_friendly","natural_light"],
     "wifi":False,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":True,"insider_note":"Magical during jacaranda season (April-May)."},
    {"name":"Kerckhoff Patio","category":"outdoor","lat":34.0713,"lng":-118.4438,
     "noise_level":"low_moderate","features":["outdoor","shaded","coffee_adjacent"],
     "wifi":False,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":True,"insider_note":"Go early — premium tables fill fast."},
    {"name":"Inverted Fountain Area","category":"outdoor","lat":34.0695,"lng":-118.4419,
     "noise_level":"quiet","features":["outdoor","shaded","benches","calming"],
     "wifi":False,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":True,"insider_note":"Most people walk past rather than stop."},
    {"name":"Lu Valle Commons Outdoor","category":"outdoor","lat":34.0738,"lng":-118.4408,
     "noise_level":"quiet","features":["outdoor","shaded","private_feel","coffee_adjacent"],
     "wifi":False,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":True,"insider_note":"Genuinely hidden gem. Dodd/law courtyard very secluded."},
    {"name":"Ackerman Union Terrace","category":"outdoor","lat":34.0707,"lng":-118.4443,
     "noise_level":"moderate","features":["outdoor","shaded","central_location","printing_nearby"],
     "wifi":False,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":True,"insider_note":"Gets busy mid-day. Better morning or late afternoon."},
    {"name":"Bunche Hall Palm Court","category":"outdoor","lat":34.0748,"lng":-118.4409,
     "noise_level":"quiet","features":["outdoor_feel","sheltered","quiet","natural_light"],
     "wifi":False,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":False,"insider_note":"Feels outdoors but protected from weather. True hidden gem."},
    {"name":"Mathias Botanical Garden","category":"outdoor","lat":34.0666,"lng":-118.4404,
     "noise_level":"very_quiet","features":["outdoor","nature","peaceful","free_admission"],
     "wifi":False,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":True,"insider_note":"Not for laptop work — perfect for reading a textbook."},
    {"name":"Court of Sciences","category":"outdoor","lat":34.0685,"lng":-118.4428,
     "noise_level":"low_moderate","features":["outdoor","tables","benches","group_friendly"],
     "wifi":False,"outlets":False,"food_drink":False,"coffee":False,"food":False,
     "weather_dependent":True,"insider_note":"Best during off-peak hours. Limited shade."},
]

DATA_FILE = Path("data/raw/ucla/all_spots.json")
if DATA_FILE.exists():
    spots = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    print(f"[INFO] Loaded {len(spots)} spots from {DATA_FILE}")
else:
    spots = INLINE_DATA
    print(f"[INFO] Using inline data ({len(spots)} spots).")

df = pd.DataFrame(spots)

# Ensure boolean columns are actually bool (not mixed with NaN)
bool_cols = ["wifi", "outlets", "food_drink", "coffee", "food", "weather_dependent"]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].fillna(False).astype(bool)

print(f"\n{'='*60}")
print("  UCLA STUDY SPOTS — EDA")
print(f"{'='*60}")
print(f"Shape : {df.shape}")
print(f"Cols  : {list(df.columns)}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — KEY VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

cat_colors = {"library": UCLA_BLUE, "study_lounge": UCLA_GOLD,
              "cafe": "#E03C31", "outdoor": "#4CAF50"}

# ── 2A: Spots per category ────────────────────────────────────────────────────
cat_counts = df["category"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("UCLA Study Spots — Overview", fontsize=15, fontweight="bold", color=UCLA_DARK)

colors = [UCLA_BLUE, UCLA_GOLD, UCLA_DARK, UCLA_LIGHT]
bars = axes[0].bar(cat_counts.index, cat_counts.values, color=colors[:len(cat_counts)])
axes[0].set_title("Spots per Category")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Count")
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(int(bar.get_height())), ha="center", fontweight="bold")

axes[1].pie(cat_counts.values, labels=cat_counts.index, colors=colors[:len(cat_counts)],
            autopct="%1.0f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Category Distribution")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_category_overview.png", bbox_inches="tight")
plt.close()
print("[VIZ] 01_category_overview.png saved")

# ── 2B: Feature heatmap ───────────────────────────────────────────────────────
ALL_FEATURES = sorted({f for row in df["features"].dropna() for f in row})

feat_matrix = pd.DataFrame(
    {feat: df["features"].apply(lambda lst: int(feat in (lst or []))) for feat in ALL_FEATURES}
)
feat_matrix["category"] = df["category"].values
feat_by_cat = feat_matrix.groupby("category")[ALL_FEATURES].mean()

fig, ax = plt.subplots(figsize=(15, 5))
sns.heatmap(feat_by_cat, annot=True, fmt=".0%", cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "% of spots in category with feature"}, ax=ax)
ax.set_title("Feature Prevalence by Category", fontsize=14, fontweight="bold", color=UCLA_DARK)
ax.set_xlabel("")
ax.set_ylabel("Category")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_feature_heatmap.png", bbox_inches="tight")
plt.close()
print("[VIZ] 02_feature_heatmap.png saved")

# ── 2C: Noise-level distribution ─────────────────────────────────────────────
NOISE_ORDER = ["silent", "very_quiet", "quiet", "low_moderate", "moderate", "loud"]

noise_df = df[df["noise_level"].notna()].copy()
noise_counts = (noise_df["noise_level"]
                .value_counts()
                .reindex([n for n in NOISE_ORDER if n in noise_df["noise_level"].values]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Noise Level Analysis", fontsize=14, fontweight="bold", color=UCLA_DARK)

palette = sns.color_palette("Blues", len(noise_counts))[::-1]
axes[0].barh(noise_counts.index, noise_counts.values, color=palette)
axes[0].set_title("Overall Noise Distribution (spots with data)")
axes[0].set_xlabel("Number of spots")
for i, v in enumerate(noise_counts.values):
    axes[0].text(v + 0.05, i, str(v), va="center", fontweight="bold")

noise_cat = noise_df.groupby(["category", "noise_level"]).size().unstack(fill_value=0)
noise_cat = noise_cat.reindex(columns=[n for n in NOISE_ORDER if n in noise_cat.columns])
noise_cat.plot(kind="bar", ax=axes[1], colormap="Blues", edgecolor="white")
axes[1].set_title("Noise Levels by Category")
axes[1].set_xlabel("Category")
axes[1].set_ylabel("Count")
axes[1].legend(title="Noise", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_noise_levels.png", bbox_inches="tight")
plt.close()
print("[VIZ] 03_noise_levels.png saved")

# ── 2D: Campus map ────────────────────────────────────────────────────────────
cat_markers = {"library": "s", "study_lounge": "D", "cafe": "o", "outdoor": "^"}

fig, ax = plt.subplots(figsize=(10, 9))
for cat, grp in df.groupby("category"):
    ax.scatter(grp["lng"], grp["lat"],
               c=cat_colors[cat], marker=cat_markers[cat],
               s=110, alpha=0.85, edgecolors="white", linewidths=0.8,
               label=cat.replace("_", " ").title(), zorder=5)
    for _, row in grp.iterrows():
        ax.annotate(row["name"], (row["lng"], row["lat"]),
                    fontsize=5.5, ha="left",
                    xytext=(4, 4), textcoords="offset points",
                    color="#333333")

ax.set_title("UCLA Campus Study Spots — Geographic Distribution",
             fontsize=13, fontweight="bold", color=UCLA_DARK)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(title="Category", frameon=True, framealpha=0.9)
ax.set_facecolor("#f0f4f8")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_campus_map.png", bbox_inches="tight")
plt.close()
print("[VIZ] 04_campus_map.png saved")

# ── 2E: Amenity availability ─────────────────────────────────────────────────
bool_summary = df[bool_cols].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(bool_summary.index, bool_summary.values * 100,
              color=[UCLA_BLUE if v > 0.5 else UCLA_LIGHT for v in bool_summary.values],
              edgecolor="white", linewidth=1.2)
ax.set_title("Amenity Availability Across All Spots (%)", fontsize=13, fontweight="bold", color=UCLA_DARK)
ax.set_ylabel("% of spots")
ax.set_ylim(0, 105)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.0f}%", ha="center", fontsize=9, fontweight="bold")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_amenity_availability.png", bbox_inches="tight")
plt.close()
print("[VIZ] 05_amenity_availability.png saved")

# ── 2F: Feature frequency ─────────────────────────────────────────────────────
feat_freq = feat_matrix[ALL_FEATURES].sum().sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(11, 5))
pal = sns.color_palette("Blues_d", len(feat_freq))
ax.bar(feat_freq.index, feat_freq.values, color=pal, edgecolor="white")
ax.set_title("Top 20 Feature Tags — Frequency Across All Spots",
             fontsize=13, fontweight="bold", color=UCLA_DARK)
ax.set_xlabel("Feature")
ax.set_ylabel("Count of spots")
plt.xticks(rotation=40, ha="right")
for i, v in enumerate(feat_freq.values):
    ax.text(i, v + 0.15, str(v), ha="center", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_feature_frequency.png", bbox_inches="tight")
plt.close()
print("[VIZ] 06_feature_frequency.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PATTERNS & INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  SECTION 3 — PATTERNS DISCOVERED")
print(f"{'='*60}")

wifi_by_cat = df.groupby("category")["wifi"].mean()
print("\n[PATTERN 1] WiFi availability by category:")
for cat, pct in wifi_by_cat.items():
    print(f"  {cat:20s}  {pct*100:.0f}%")

outlets_by_cat = df.groupby("category")["outlets"].mean()
print("\n[PATTERN 2] Outlet availability by category:")
for cat, pct in outlets_by_cat.items():
    print(f"  {cat:20s}  {pct*100:.0f}%")

print("\n[PATTERN 3] Noise level profile (spots with data):")
print(noise_counts.to_string())

weather_pct = df["weather_dependent"].mean()
print(f"\n[PATTERN 4] Weather-dependent spots: {df['weather_dependent'].sum()}/{len(df)} "
      f"({weather_pct*100:.0f}%)")

df["campus_zone"] = pd.cut(df["lat"],
    bins=[34.0660, 34.0710, 34.0740, 34.0760],
    labels=["South Campus", "Central Campus", "North Campus"])
print("\n[PATTERN 5] Geographic clustering:")
print(df["campus_zone"].value_counts().to_string())

df["feature_count"] = df["features"].apply(lambda x: len(x) if x else 0)

# ── Correlation heatmap ───────────────────────────────────────────────────────
num_cols = ["wifi", "outlets", "food_drink", "coffee", "food", "weather_dependent"]
corr = df[num_cols].astype(int).corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Matrix — Amenity Attributes",
             fontsize=12, fontweight="bold", color=UCLA_DARK)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_correlation_matrix.png", bbox_inches="tight")
plt.close()
print("[VIZ] 07_correlation_matrix.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA QUALITY AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  SECTION 4 — DATA QUALITY AUDIT")
print(f"{'='*60}")

missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(1)
quality_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
quality_df = quality_df[quality_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
print("\n[QUALITY 1] Missing / null values per column:")
print(quality_df.to_string())

missing_noise = df["noise_level"].isna().sum()
print(f"\n[QUALITY 2] noise_level missing for {missing_noise}/{len(df)} spots "
      f"({missing_noise/len(df)*100:.0f}%) — all libraries.")

dup_coords = df.groupby(["lat","lng"]).filter(lambda g: len(g) > 1)[["name","lat","lng"]]
if len(dup_coords):
    print(f"\n[QUALITY 3] {len(dup_coords)} spots share a lat/lng with another spot:")
    print(dup_coords.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
q_cols = [c for c in df.columns if df[c].isna().sum() > 0]
q_vals = [df[c].isna().mean()*100 for c in q_cols]
color_q = [UCLA_BLUE if v < 30 else UCLA_GOLD if v < 60 else "#E03C31" for v in q_vals]
ax.barh(q_cols, q_vals, color=color_q)
ax.set_title("Data Quality — % Missing per Column", fontsize=13, fontweight="bold", color=UCLA_DARK)
ax.set_xlabel("% missing")
ax.axvline(x=50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_data_quality.png", bbox_inches="tight")
plt.close()
print("[VIZ] 08_data_quality.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FEATURE ENGINEERING (study_score + dist_to_nearest_cafe_m only)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  SECTION 5 — FEATURE ENGINEERING")
print(f"{'='*60}")

# ── 5.1 study_score ──────────────────────────────────────────────────────────
NOISE_SCORE = {
    "silent": 5, "very_quiet": 4, "quiet": 3,
    "low_moderate": 2, "moderate": 1, "loud": 0,
}

def compute_study_score(row):
    score = 0
    score += 2 if row["wifi"] else 0
    score += 2 if row["outlets"] else 0
    score += NOISE_SCORE.get(row["noise_level"], 2)   # unknown noise → neutral (2)
    score += 1 if row["food_drink"] else 0
    score += min(row["feature_count"] / 3, 2)          # feature richness, capped at 2
    score -= 1 if row["weather_dependent"] else 0
    return round(score, 2)

df["study_score"] = df.apply(compute_study_score, axis=1)

print("\n[FEATURE] study_score — top 10:")
print(df[["name", "category", "study_score"]]
      .sort_values("study_score", ascending=False)
      .head(10).to_string(index=False))

# ── 5.2 dist_to_nearest_cafe_m ───────────────────────────────────────────────
cafe_coords = df[df["category"] == "cafe"][["lat", "lng"]].values

def haversine_min(lat, lng, targets):
    """Approximate great-circle distance in metres to nearest target."""
    R = 6_371_000
    lat_r = np.radians(lat)
    lng_r = np.radians(lng)
    t_lat = np.radians(targets[:, 0])
    t_lng = np.radians(targets[:, 1])
    dlat = t_lat - lat_r
    dlng = t_lng - lng_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(t_lat) * np.sin(dlng / 2) ** 2
    return float(np.min(R * 2 * np.arcsin(np.sqrt(a))))

df["dist_to_nearest_cafe_m"] = df.apply(
    lambda r: haversine_min(r["lat"], r["lng"], cafe_coords), axis=1
).round(0)

print("\n[FEATURE] dist_to_nearest_cafe_m — farthest 8 spots:")
print(df[["name", "category", "dist_to_nearest_cafe_m"]]
      .sort_values("dist_to_nearest_cafe_m", ascending=False)
      .head(8).to_string(index=False))

# ── Study score visualisation ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
sorted_df = df.sort_values("study_score", ascending=True)
bar_colors = [cat_colors[c] for c in sorted_df["category"]]
ax.barh(sorted_df["name"], sorted_df["study_score"], color=bar_colors, edgecolor="white")
ax.set_title("Study Score by Spot (composite metric)", fontsize=13, fontweight="bold", color=UCLA_DARK)
ax.set_xlabel("Study Score (higher = better)")
patches = [mpatches.Patch(color=v, label=k.replace("_", " ").title()) for k, v in cat_colors.items()]
ax.legend(handles=patches, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_study_scores.png", bbox_inches="tight")
plt.close()
print("[VIZ] 09_study_scores.png saved")

# ── Distance to nearest cafe visualisation ────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
sorted_dist = df.sort_values("dist_to_nearest_cafe_m", ascending=True)
dist_colors = [cat_colors[c] for c in sorted_dist["category"]]
ax.barh(sorted_dist["name"], sorted_dist["dist_to_nearest_cafe_m"], color=dist_colors, edgecolor="white")
ax.set_title("Distance to Nearest Cafe (metres)", fontsize=13, fontweight="bold", color=UCLA_DARK)
ax.set_xlabel("Distance (m)")
patches = [mpatches.Patch(color=v, label=k.replace("_", " ").title()) for k, v in cat_colors.items()]
ax.legend(handles=patches, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_dist_to_nearest_cafe.png", bbox_inches="tight")
plt.close()
print("[VIZ] 10_dist_to_nearest_cafe.png saved")

# ── Save enriched dataset ─────────────────────────────────────────────────────
out_csv = OUTPUT_DIR / "ucla_study_spots_enriched.csv"
df.to_csv(out_csv, index=False)
print(f"\n[OUTPUT] Enriched dataset saved → {out_csv}")

print(f"\n{'='*60}")
print("  EDA COMPLETE")
print(f"  Outputs in: {OUTPUT_DIR.resolve()}")
print(f"{'='*60}\n")