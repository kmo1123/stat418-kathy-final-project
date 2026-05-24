"""
app.py  —  UCLA Study Spot Recommender  (Streamlit)
-----------------------------------------------------
Talks to the FastAPI backend at API_URL.
Run:
  # Terminal 1 — start the API
  uvicorn api:app --port 8000

  # Terminal 2 — start the dashboard
  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import os
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Locally:   set API_URL in .env or just leave as localhost default
# Deployed:  set API_URL in Streamlit Cloud secrets (Settings → Secrets)
#            as:  API_URL = "https://your-api-xyz.run.app"
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

CATEGORY_EMOJI = {
    "library":      "📚",
    "study_lounge": "🛋️",
    "cafe":         "☕",
    "outdoor":      "🌿",
}
CATEGORY_COLOR = {
    "library":      "#2B6CB0",
    "study_lounge": "#2D7D46",
    "cafe":         "#C05621",
    "outdoor":      "#6B46C1",
}
NOISE_LABEL = {
    "silent":       "🔇 Silent",
    "very_quiet":   "🤫 Very Quiet",
    "quiet":        "🔈 Quiet",
    "low_moderate": "🔉 Low–Moderate",
    "moderate":     "🔊 Moderate",
    None:           "—",
}

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_all_spots():
    try:
        r = requests.get(f"{API_URL}/spots", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def fetch_recommendations(prefs: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/recommend", json=prefs, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def fetch_explanation(spot_name: str, prefs: dict) -> list | None:
    try:
        r = requests.post(
            f"{API_URL}/recommend/explain",
            json={"spot_name": spot_name, "prefs": prefs},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()["contributions"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UCLA Study Spot Finder",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Tighten card padding */
  .spot-card {
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    border-left: 5px solid #ccc;
  }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px 2px 2px 0;
    color: white;
  }
  .sim-bar-bg {
    background: #e2e8f0;
    border-radius: 6px;
    height: 8px;
    margin-top: 4px;
  }
  .sim-bar-fill {
    border-radius: 6px;
    height: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — user preferences
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://brand.ucla.edu/images/logos/ucla-logo.png", width=140)
    st.title("Find Your Spot")
    st.caption("Tell us what you need and we'll find the best matches.")
    st.divider()

    st.subheader("📶 Connectivity")
    wants_wifi    = st.toggle("WiFi",    value=True)
    wants_outlets = st.toggle("Outlets", value=False)

    st.subheader("🍽️ Food & Coffee")
    wants_coffee = st.toggle("Coffee nearby", value=False)
    wants_food   = st.toggle("Food nearby",   value=False)

    st.subheader("🔇 Environment")
    wants_quiet   = st.toggle("Need it quiet",     value=False)
    wants_outdoor = st.toggle("Prefer outdoor",    value=False)
    wants_comfortable = st.toggle("Comfortable seating", value=False)

    st.subheader("👥 Session Type")
    wants_group = st.toggle("Group study", value=False)

    st.subheader("⚙️ Other")
    weather_ok = st.toggle(
        "OK with weather-dependent spots",
        value=True,
        help="Outdoor spots may not be usable when it rains.",
    )
    noise_tolerance = st.select_slider(
        "Noise tolerance",
        options=["Silent", "Very Quiet", "Quiet", "Low–Moderate", "Moderate"],
        value="Quiet",
    )
    connectivity_need = st.select_slider(
        "How critical is connectivity?",
        options=["Not at all", "Nice to have", "Important", "Critical"],
        value="Nice to have",
    )
    n_results = st.slider("Number of results", 1, 10, 3)

    st.divider()
    find_btn = st.button("🔍 Find Study Spots", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Map noise/connectivity sliders to numeric values
# ---------------------------------------------------------------------------

NOISE_MAP = {
    "Silent": 0, "Very Quiet": 1, "Quiet": 2, "Low–Moderate": 3, "Moderate": 4
}
CONN_MAP = {
    "Not at all": 0, "Nice to have": 1, "Important": 2, "Critical": 3
}

prefs = {
    "wants_wifi":        int(wants_wifi),
    "wants_outlets":     int(wants_outlets),
    "wants_coffee":      int(wants_coffee),
    "wants_food":        int(wants_food),
    "wants_quiet":       int(wants_quiet),
    "wants_outdoor":     int(wants_outdoor),
    "wants_group":       int(wants_group),
    "wants_comfortable": int(wants_comfortable),
    "weather_ok":        int(weather_ok),
    "noise_tolerance":   NOISE_MAP[noise_tolerance],
    "connectivity_need": CONN_MAP[connectivity_need],
    "n_results":         n_results,
}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_recs, tab_browse, tab_about = st.tabs(
    ["🎯 Recommendations", "🗺️ Browse All Spots", "ℹ️ About"]
)

# ===========================
# TAB 1 — RECOMMENDATIONS
# ===========================

with tab_recs:
    st.header("Your Top Study Spots")

    if not find_btn and "last_results" not in st.session_state:
        st.info("Set your preferences in the sidebar and click **Find Study Spots**.")
    else:
        if find_btn:
            with st.spinner("Finding the best spots for you..."):
                data = fetch_recommendations(prefs)
            if data:
                st.session_state["last_results"] = data
                st.session_state["last_prefs"]   = prefs

        data  = st.session_state.get("last_results")
        prefs_used = st.session_state.get("last_prefs", prefs)

        if not data:
            st.warning("No results returned. Make sure the API is running.")
        else:
            results = data["results"]
            st.caption(f"Showing {len(results)} recommendation{'s' if len(results) > 1 else ''} "
                       f"based on your preferences.")

            # ---- Result cards ----
            for spot in results:
                cat   = spot["category"]
                color = CATEGORY_COLOR.get(cat, "#888")
                emoji = CATEGORY_EMOJI.get(cat, "📍")
                sim   = spot["similarity"]
                noise = NOISE_LABEL.get(spot.get("noise_level"), "—")

                with st.container():
                    # Rank + match bar
                    left, right = st.columns([8, 2])
                    with left:
                        st.markdown(
                            f"### {spot['rank']}. {emoji} {spot['name']}",
                            unsafe_allow_html=False,
                        )
                        st.caption(
                            f"{cat.replace('_',' ').title()}"
                            + (f" · {spot['building']}" if spot.get("building") else "")
                            + (f" · {spot['area']}"     if spot.get("area")     else "")
                        )
                    with right:
                        pct = int(sim * 100)
                        st.metric("Match", f"{pct}%")

                    # Description
                    if spot.get("description"):
                        st.write(spot["description"])

                    # Feature badges
                    feats = spot.get("features") or []
                    if feats:
                        badge_html = " ".join(
                            f'<span class="badge" style="background:{color};">'
                            f'{f.replace("_"," ").title()}</span>'
                            for f in feats
                        )
                        st.markdown(badge_html, unsafe_allow_html=True)

                    # Info columns
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**Noise level:** {noise}")
                        if spot.get("hours"):
                            st.markdown(f"**Hours:** {spot['hours']}")
                        if spot.get("hours_note"):
                            st.caption(spot["hours_note"])
                        if spot.get("access"):
                            st.markdown(f"**Access:** {spot['access']}")
                    with c2:
                        if spot.get("best_times"):
                            st.markdown("**Best times:**")
                            for t in spot["best_times"]:
                                st.markdown(f"  ✅ {t}")
                    with c3:
                        if spot.get("avoid_times"):
                            st.markdown("**Avoid:**")
                            for t in spot["avoid_times"]:
                                st.markdown(f"  ⚠️ {t}")
                        if spot.get("food_nearby"):
                            st.markdown(f"**Food nearby:** {spot['food_nearby']}")

                    # Links
                    link_cols = st.columns(3)
                    with link_cols[0]:
                        if spot.get("url"):
                            st.link_button("🔗 More info", spot["url"])
                    with link_cols[1]:
                        if spot.get("lat") and spot.get("lng"):
                            maps_url = (
                                f"https://www.google.com/maps/search/?api=1"
                                f"&query={spot['lat']},{spot['lng']}"
                            )
                            st.link_button("🗺️ Google Maps", maps_url)

                    # Why this spot? (expandable explanation)
                    with st.expander("🔍 Why was this recommended?"):
                        contribs = fetch_explanation(spot["name"], prefs_used)
                        if contribs:
                            import pandas as pd
                            df_exp = pd.DataFrame(contribs[:8])
                            df_exp = df_exp.rename(columns={
                                "feature":      "Feature",
                                "user_value":   "Your Pref",
                                "spot_value":   "Spot Value",
                                "rf_weight":    "Signal Weight",
                                "contribution": "Match Contribution",
                            })
                            st.dataframe(
                                df_exp.style.background_gradient(
                                    subset=["Match Contribution"], cmap="Greens"
                                ).format(precision=3),
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.caption(
                                "Signal Weight = how much the model relies on this feature. "
                                "Match Contribution = how much this feature drove the match."
                            )
                        else:
                            st.caption("Explanation unavailable — is the API running?")

                    st.divider()

# ===========================
# TAB 2 — BROWSE ALL SPOTS
# ===========================

with tab_browse:
    st.header("All UCLA Study Spots")

    spots = fetch_all_spots()
    if not spots:
        st.warning("Could not load spots. Make sure the API is running.")
    else:
        # Filter controls
        fc1, fc2 = st.columns(2)
        with fc1:
            cat_filter = st.multiselect(
                "Filter by category",
                options=["library", "study_lounge", "cafe", "outdoor"],
                default=["library", "study_lounge", "cafe", "outdoor"],
                format_func=lambda c: f"{CATEGORY_EMOJI.get(c,'')} {c.replace('_',' ').title()}",
            )
        with fc2:
            noise_filter = st.multiselect(
                "Filter by noise level",
                options=["silent", "very_quiet", "quiet", "low_moderate", "moderate"],
                default=["silent", "very_quiet", "quiet", "low_moderate", "moderate"],
                format_func=lambda n: NOISE_LABEL.get(n, n),
            )

        wifi_only    = st.checkbox("WiFi only")
        outlets_only = st.checkbox("Outlets only")
        outdoor_only = st.checkbox("Outdoor / weather-independent only")

        filtered = [
            s for s in spots
            if s["category"] in cat_filter
            and s.get("noise_level", "moderate") in noise_filter
            and (not wifi_only    or s.get("wifi") or "wifi" in (s.get("features") or []))
            and (not outlets_only or s.get("outlets") or "outlets" in (s.get("features") or []))
            and (not outdoor_only or s.get("weather_dependent") is False)
        ]

        st.caption(f"Showing {len(filtered)} of {len(spots)} spots")

        # Map
        if any(s.get("lat") for s in filtered):
            import pandas as pd
            map_df = pd.DataFrame([
                {"lat": s["lat"], "lon": s["lng"], "name": s["name"]}
                for s in filtered if s.get("lat") and s.get("lng")
            ])
            st.map(map_df, latitude="lat", longitude="lon", size=40)

        # Spot grid
        cols = st.columns(2)
        for i, spot in enumerate(filtered):
            cat   = spot["category"]
            color = CATEGORY_COLOR.get(cat, "#888")
            emoji = CATEGORY_EMOJI.get(cat, "📍")
            noise = NOISE_LABEL.get(spot.get("noise_level"), "—")

            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"#### {emoji} {spot['name']}")
                    st.caption(
                        cat.replace("_", " ").title()
                        + (f" · {spot.get('building','')}" if spot.get("building") else "")
                        + (f" · {spot.get('area','')}"     if spot.get("area")     else "")
                    )
                    if spot.get("description"):
                        st.write(spot["description"][:180] + ("…" if len(spot.get("description","")) > 180 else ""))

                    feats = spot.get("features") or []
                    if feats:
                        badge_html = " ".join(
                            f'<span class="badge" style="background:{color};">'
                            f'{f.replace("_"," ").title()}</span>'
                            for f in feats[:6]
                        )
                        st.markdown(badge_html, unsafe_allow_html=True)

                    st.markdown(f"**Noise:** {noise}")
                    if spot.get("hours"):
                        st.caption(f"Hours: {spot['hours']}")
                    if spot.get("url"):
                        st.link_button("More info →", spot["url"])

# ===========================
# TAB 3 — ABOUT
# ===========================

with tab_about:
    st.header("How It Works")

    st.markdown("""
    ### The Model: RF-Weighted KNN

    This recommender uses **content-based filtering** — it represents both study spots
    and your preferences as vectors in a shared feature space, then finds the spots
    closest to your preference vector.

    Two things make it smarter than plain nearest-neighbor search:

    1. **Random Forest feature weighting** — A Random Forest is trained to classify
       spots into categories (library, cafe, lounge, outdoor) from their features.
       The feature importances it learns are used to *weight* each dimension in the
       similarity calculation. Features that are more discriminating (like amenity score,
       WiFi presence, and whether a spot is outdoor) pull more weight; low-signal tags
       (like "benches" or "shaded") contribute less.

    2. **Cosine similarity** — Measures the angle between your preference vector and
       each spot vector, not raw distance. This means a spot with 2 matching features
       out of 2 scores higher than one with 2 matching features out of 10, which is
       the right behavior for preference matching.

    ### Why Not ROC / Accuracy / F1?

    Those are **classification** metrics — they measure how well a model predicts a
    discrete label (e.g. "is this a library?"). This problem is **retrieval**: given a
    query, rank 30 items by relevance. The metrics used were:

    | Metric | What it measures |
    |--------|-----------------|
    | **Precision@3** | Of the 3 spots returned, what fraction are in the same category as the query? |
    | **MRR** | Mean Reciprocal Rank — how early does the first relevant result appear? |

    RF-KNN scored **0.967** on both metrics (vs 0.689 for Jaccard, the worst performer).

    ### Feature Engineering Summary

    | Decision | Rationale |
    |----------|-----------|
    | Dropped 15 singleton tags | Appear in only 1 spot — zero discriminative power |
    | Dropped 8 duplicate features | `feat_coffee == feat_food` etc. — would double-weight artificially |
    | Ordinal-encoded noise level | Preserves order (silent < quiet < moderate); OHE would lose it |
    | Composite scores (comfort, connectivity, group, outdoor) | Summarize correlated tag clusters; cleaner signal for RF |
    | Imputed library noise with median | Libraries have no noise_level from scraping; median keeps them in the pool |

    ### Data Sources

    | Category | Source | Method |
    |----------|--------|--------|
    | Libraries | library.ucla.edu | Scraped (requests + BeautifulSoup) |
    | Study Lounges | asucla.ucla.edu | Static (JS-rendered, not scrapable) |
    | Cafes | asucla.ucla.edu, dining.ucla.edu | Static (JS-rendered, not scrapable) |
    | Outdoor | Curated | Static |
    """)

    st.divider()
    st.caption("Built with FastAPI · Scikit-learn · Streamlit")