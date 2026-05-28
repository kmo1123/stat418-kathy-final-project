# UCLA Study Spot Recommender

A content-based recommendation system that helps UCLA students find the right on-campus study spot based on their preferences. Built with a FastAPI backend, a Streamlit frontend, and an RF-weighted KNN model.

**App:** `https://stat418-kathy-final-project-bmpv6hxtgtpelge29ujkqn.streamlit.app/ `  
**API:** `https://ucla-study-api-853468627492.us-central1.run.app/docs`  

## Project Overview

UCLA has over 30 distinct study spots across libraries, cafes, lounges, and outdoor areas — each with different noise levels, amenities, and best use cases. This project collects structured data on all of them, trains a recommender model, and serves recommendations through a REST API that a Streamlit dashboard calls in real time.

The user sets preferences (quiet vs. noisy, needs wifi/outlets, indoor vs. outdoor, solo vs. group) and receives three ranked recommendations with match scores, best times to visit, and a per-feature explanation of why each spot was suggested.

## Solution Architecture

<img width="874" height="1308" alt="Screenshot 2026-05-26 at 7 36 02 PM" src="https://github.com/user-attachments/assets/6e44c2bc-1ec4-48b6-b6a1-00ca761db734" />

## Repository Structure

```
ucla-study-spots/
├── api.py                          # FastAPI backend
├── app.py                          # Streamlit frontend
├── ucla_collector.py               # Data collection pipeline
├── eda.py                          # EDA and feature engineering
├── model.py                        # Model training and evaluation
├── dockerfile                      # Docker container for the API
├── requirements.txt                # All dependencies
├── data/
│   └── raw/ucla/
│       ├── all_spots.json          # Combined spot data (30 spots)
│       ├── libraries/              # Per-library JSON files
│       ├── cafes/                  # Per-cafe JSON files
│       ├── lounges/                # Per-lounge JSON files
│       └── outdoor/                # Per-outdoor-spot JSON files
├── tables/
│   ├── engineered_df.csv           # Feature-engineered dataset
│   ├── feature_matrix.csv          # OHE feature presence matrix
│   ├── model_scores.csv            # Model comparison results
│   ├── summary_stats.csv           # Per-category statistics
│   └── recommendations_demo.csv    # Sample recommendation outputs
└── plots/                          # EDA and model visualization outputs
```

## Setup and Installation

### Prerequisites

- Python 3.11+
- pip
- (For API deployment) Docker or Podman
- (For deployment) Google Cloud account, Docker Hub account, Streamlit account

### 1. Clone the repository

```bash
git clone https://github.com/kmo1123/stat418-kathy-final-project.git
cd stat418-kathy-final-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Collect data

```bash
python ucla_study_spots_collector.py
```

This scrapes the UCLA library website and loads static data for cafes, lounges, and outdoor spots. Outputs go to `data/raw/ucla/`.

### 4. Run EDA and feature engineering

```bash
python ucla_study_spots_eda.py
```

Outputs plots to `plots/` and engineered features to `tables/engineered_df.csv`.

### 5. Train and evaluate the model

```bash
python ucla_study_spots_model.py
```

Outputs model comparison scores to `tables/model_scores.csv` and demo recommendations to `tables/recommendations_demo.csv`.

### 6. Run locally

Open two terminal windows in the project directory:

```bash
# Terminal 1 — start the API
uvicorn api:app --reload --port 8000

# Terminal 2 — start the Streamlit app
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.  

## Model

### Problem framing

With 30 spots this is a **content-based filtering** problem, not supervised classification. The goal is to map a user preference vector into the same feature space as the spots and retrieve the nearest neighbors.

### Feature engineering decisions

| Decision | Rationale |
|----------|-----------|
| Dropped 15 singleton feature tags | Appear in only 1 spot — zero discriminative power |
| Dropped 8 duplicate features | `feat_coffee == feat_food` (always co-occur in cafes); keeping duplicates inflates their effective weight |
| Ordinal-encoded noise level | Preserves silent < quiet < moderate ordering; OHE would lose it |
| Composite scores (comfort, connectivity, group, outdoor) | Summarize correlated tag clusters into cleaner signals |
| Imputed library noise with category median | Libraries have no `noise_level` from scraping; median keeps them in the pool |

### Model comparison

| Model | Precision@3 | MRR |
|-------|------------|-----|
| **RF-KNN** | **0.967** | **0.967** |
| Plain KNN | 0.956 | 0.950 |
| SVD | 0.956 | 0.950 |
| Jaccard | 0.689 | 0.683 |

### How RF-KNN works

A Random Forest is trained to classify spots by category. Its feature importances are used to weight each dimension in the KNN similarity calculation. Features that better distinguish spot types pull more weight in the distance calculation. This outperforms plain cosine KNN because it down-weights low-signal tags and up-weights the features that actually matter (amenity score, outdoor/indoor, noise level, wifi).

## API Endpoints

The API runs on FastAPI and auto-generates interactive documentation at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check — confirms model is loaded |
| GET | `/stats` | Request counts and uptime |
| GET | `/spots` | All 30 spots with display metadata |
| POST | `/recommend` | Top-N recommendations given user preferences |
| POST | `/recommend/explain` | Per-feature contribution breakdown for a spot |

### Example request

```bash
curl -X POST https://your-api.run.app/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "wants_wifi": 1,
    "wants_outlets": 1,
    "wants_quiet": 1,
    "noise_tolerance": 1,
    "n_results": 3
  }'
```

### Example response

```json
{
  "results": [
    {
      "rank": 1,
      "name": "Louise Kerckhoff Study Lounge",
      "category": "study_lounge",
      "similarity": 0.9115,
      "noise_level": "quiet",
      "features": ["wifi", "outlets", "comfortable_seating", "quiet"],
      "best_times": ["weekday mornings"],
      "avoid_times": ["midterms and finals week afternoons"]
    }
  ],
  "latency_ms": 3.2
}
```

## Deployment

### API — Google Cloud Run

```bash
# Build for linux/amd64 (required for Cloud Run, even on Mac)
podman build --platform linux/amd64 -f dockerfile -t yourusername/ucla-study-api:latest .

# Push to Docker Hub
podman login docker.io
podman push yourusername/ucla-study-api:latest
```

Then in Google Cloud Run: Create Service → container image URL → set port 8080 → Allow unauthenticated invocations → Create.

### App — Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set main file to `app.py`
4. Under Settings → Secrets, add:

```toml
API_URL = "https://your-api.run.app"
```

5. Deploy

## AI Assistant Documentation

### Tools used

**Claude (Anthropic)** was the primary AI assistant used throughout this project, accessed via claude.ai.

### Tasks where AI assistance was used

| Task | How AI was used |
|------|----------------|
| FastAPI structure | Generated the initial endpoint structure, Pydantic models, and lifespan handler |
| Streamlit layout | Scaffolded the three-tab layout, sidebar controls, result cards, and the explanation table |
| Deployment | Explained the Docker → Docker Hub → Cloud Run pipeline, the difference between `streamlit run` and `python`, and how to configure Streamlit secrets for the API URL |

### Examples of helpful prompts

**Debugging the deployment connection error:**
> "API error: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded"

Immediately identified the cause (Streamlit Cloud can't reach localhost) and explained that the API URL needed to be set in Streamlit secrets rather than hardcoded.


### Challenges where AI assistance was most valuable

**Understanding why the Streamlit app wasn't launching.**  
Running `python app.py` instead of `streamlit run app.py` produced a wall of `ScriptRunContext` warnings with no browser window. Without knowing the correct launch command the error messages were not useful on their own.

**Structuring the two-deployment architecture.**  
The relationship between Docker, Docker Hub, Cloud Run, and Streamlit Community Cloud and why they are separate deployments rather than one was explained clearly when the distinction between containerized API hosting and GitHub-connected app hosting was not obvious.


### Areas where AI-generated code was significantly modified

**The `spaces` field in library records.**  
The scraper included a `spaces` section that parsed for an `h2` element called "Spaces" on the library pages. In practice this element never appeared, so the field was always an empty array in every output. It was removed from the schema entirely once this was caught by inspecting the actual output.

**The `--maps` flag and Playwright code.**  
A Playwright-based Google Maps scraper for popular times data was written and included in the collector. After testing, it became clear the scraper was not actually retrieving any data (the Maps site blocks headless browsers), and the flag was silently doing nothing. The entire Playwright section was removed rather than left as dead code.

### Lessons learned about working with AI coding assistants

**Verify output before assuming it works.**  
Several features were generated that looked correct but produced no real output, such as the Playwright Maps scraper and the `spaces` scraper. Running the code and inspecting the actual JSON output revealed these gaps in a way that reading the code alone did not.

**Incremental correction works well.**  
Rather than trying to get everything right in one pass, iterating was more reliable than asking for complete rewrites. Most bugs were fixed by describing the exact error message and getting a precise change rather than regenerating large sections.
