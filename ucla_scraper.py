"""
ucla_study_spots_collector.py
------------------------------
Collects UCLA on-campus study spot data. Covers four categories:

  1. Libraries     — scraped from library.ucla.edu (8 branches)
  2. Study Lounges — static data from asucla.ucla.edu (6 lounges)
  3. Cafes         — static data; ASUCLA site is JS-rendered so not scrapable (8 cafes)
  4. Outdoor       — static curated data (8 spots)

Fields per spot:
  - Scraped where possible: name, description, address, phone, email,
    amenities, features, hours_url
  - Static/curated: lat, lng, building, noise_level, food_drink,
    access, hours (cafes only), indoor_outdoor (cafes only)
  - Timing estimates: best_times, avoid_times
    NOTE — these are general estimates based on typical university patterns,
    not verified against live traffic data. They can change term to term.

Outputs:
  data/raw/ucla/libraries/     — one JSON per library
  data/raw/ucla/lounges/       — one JSON per lounge
  data/raw/ucla/cafes/         — one JSON per cafe
  data/raw/ucla/outdoor/       — one JSON per outdoor spot
  data/raw/ucla/all_spots.json — combined master file

Usage:
  pip install requests beautifulsoup4
  python ucla_scraper.py
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent
LOG_DIR     = BASE_DIR / "logs"
DATA_DIR    = BASE_DIR / "data" / "raw" / "ucla"
LIBRARY_DIR = DATA_DIR / "libraries"
LOUNGE_DIR  = DATA_DIR / "lounges"
CAFE_DIR    = DATA_DIR / "cafes"
OUTDOOR_DIR = DATA_DIR / "outdoor"

for d in [LOG_DIR, LIBRARY_DIR, LOUNGE_DIR, CAFE_DIR, OUTDOOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    filename=str(LOG_DIR / "ucla_collector.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

DELAY = 2.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TIMES_DISCLAIMER = (
    "General estimate based on typical university patterns. "
    "Actual occupancy varies by term, week, and day."
)

# ---------------------------------------------------------------------------
# Library metadata
# Fields scraped from library.ucla.edu: description, address, phone, email,
# amenities, features. Fields below are static (coordinates, timing).
# ---------------------------------------------------------------------------

LIBRARY_PAGES = [
    {
        "name": "Powell Library",
        "url": "https://www.library.ucla.edu/visit/locations/powell-library/",
        "lat": 34.0713, "lng": -118.4417,
        "best_times": ["weekday mornings before 10am", "weekday evenings after 8pm"],
        "avoid_times": ["weekdays 11am-3pm during midterms and finals"],
    },
    {
        "name": "Charles E. Young Research Library (YRL)",
        "url": "https://www.library.ucla.edu/visit/locations/research-library/",
        "lat": 34.0752, "lng": -118.4418,
        "best_times": ["weekday mornings", "Friday afternoons", "weekends before noon"],
        "avoid_times": ["Sunday evenings during midterms and finals"],
    },
    {
        "name": "Science and Engineering Library (SEL)",
        "url": "https://www.library.ucla.edu/visit/locations/science-engineering-library/",
        "lat": 34.0692, "lng": -118.4432,
        "best_times": ["weekdays before 11am", "evenings after 7pm", "weekends"],
        "avoid_times": ["Tuesday and Thursday afternoons"],
    },
    {
        "name": "Louise M. Darling Biomedical Library",
        "url": "https://www.library.ucla.edu/visit/locations/biomed/",
        "lat": 34.0663, "lng": -118.4446,
        "best_times": ["weekday mornings and afternoons"],
        "avoid_times": ["early afternoon during health sciences class transitions"],
    },
    {
        "name": "Walter H. Rubsamen Music Library",
        "url": "https://www.library.ucla.edu/visit/locations/music-library/",
        "lat": 34.0704, "lng": -118.4398,
        "best_times": ["any time during operating hours"],
        "avoid_times": [],
    },
    {
        "name": "Richard C. Rudolph East Asian Library",
        "url": "https://www.library.ucla.edu/visit/locations/east-asian-library/",
        "lat": 34.0752, "lng": -118.4415,
        "best_times": ["any time during operating hours"],
        "avoid_times": [],
    },
    {
        "name": "Management Library",
        "url": "https://www.library.ucla.edu/visit/locations/management-library/",
        "lat": 34.0730, "lng": -118.4397,
        "best_times": ["weekday mornings", "Friday afternoons"],
        "avoid_times": ["mid-morning and mid-afternoon during MBA class transitions"],
    },
    {
        "name": "Arts Library",
        "url": "https://www.library.ucla.edu/visit/locations/arts-library/",
        "lat": 34.0736, "lng": -118.4393,
        "best_times": ["weekday mornings and afternoons"],
        "avoid_times": [],
    },
]

# ---------------------------------------------------------------------------
# Study lounges
# ASUCLA site is JS-rendered — not scrapable with plain requests.
# Data is static, sourced from asucla.ucla.edu/study-lounges.
# ---------------------------------------------------------------------------

ASUCLA_LOUNGES = [
    {
        "name": "Louise Kerckhoff Study Lounge",
        "building": "Kerckhoff Hall, 3rd Floor",
        "lat": 34.0713, "lng": -118.4440,
        "description": (
            "Co-ed study lounge with comfortable couches, overhead lighting, "
            "and stained glass bay windows. Originally the Women's Smoking Room."
        ),
        "features": ["comfortable_seating", "quiet", "wifi", "outlets"],
        "noise_level": "quiet",
        "food_drink": False,
        "access": "UCLA students, faculty, staff",
        "hours_note": "Check asucla.ucla.edu/study-lounges for current hours.",
        "url": "https://www.asucla.ucla.edu/study-lounges",
        "best_times": ["weekday mornings"],
        "avoid_times": ["midterms and finals week afternoons"],
    },
    {
        "name": "Kerckhoff Second Floor Study Lounge",
        "building": "Kerckhoff Hall, 2nd Floor",
        "lat": 34.0713, "lng": -118.4440,
        "description": (
            "Smaller lounge one floor below the third-floor space. "
            "Adjacent to Kerckhoff Coffeehouse."
        ),
        "features": ["comfortable_seating", "quiet", "wifi"],
        "noise_level": "quiet",
        "food_drink": False,
        "access": "UCLA students, faculty, staff",
        "hours_note": "Check asucla.ucla.edu/study-lounges for current hours.",
        "url": "https://www.asucla.ucla.edu/study-lounges",
        "best_times": ["any time the third-floor lounge is full"],
        "avoid_times": [],
    },
    {
        "name": "Ackerman Union Open Study",
        "building": "Ackerman Union",
        "lat": 34.0707, "lng": -118.4440,
        "description": (
            "Unreserved meeting rooms available as open study space when not booked. "
            "Check the schedule posted at each room entrance."
        ),
        "features": ["wifi", "flexible_seating", "group_friendly", "printing_nearby"],
        "noise_level": "moderate",
        "food_drink": False,
        "access": "UCLA students, faculty, staff (unreserved rooms only)",
        "hours_note": "Varies by room. Check posted schedule.",
        "url": "https://www.asucla.ucla.edu/study-lounges",
        "best_times": ["Friday afternoons", "mornings before rooms are reserved"],
        "avoid_times": ["Monday-Thursday daytime"],
    },
    {
        "name": "Bruin Reflection Space",
        "building": "Ackerman Union, 3rd Floor",
        "lat": 34.0707, "lng": -118.4440,
        "description": (
            "Non-denominational space for contemplation, meditation, and prayer. "
            "Open to all students. Quiet atmosphere maintained."
        ),
        "features": ["quiet", "wifi", "comfortable_seating"],
        "noise_level": "silent",
        "food_drink": False,
        "access": "All students",
        "hours_note": "Check asucla.ucla.edu/study-lounges for current hours.",
        "url": "https://www.asucla.ucla.edu/study-lounges",
        "best_times": ["any time during operating hours"],
        "avoid_times": [],
    },
    {
        "name": "Transfer Student Center Lounge",
        "building": "Ackerman Union",
        "lat": 34.0707, "lng": -118.4440,
        "description": (
            "Lounge inside the UCLA Transfer Student Center. "
            "Also open to commuter and non-traditional students."
        ),
        "features": ["wifi", "comfortable_seating"],
        "noise_level": "moderate",
        "food_drink": False,
        "access": "All students (transfer-focused)",
        "hours_note": "Check Transfer Student Center for current hours.",
        "url": "https://www.asucla.ucla.edu/study-lounges",
        "best_times": ["between classes on weekdays"],
        "avoid_times": ["Transfer Center event periods"],
    },
    {
        "name": "Veteran Resource Center",
        "building": "Kerckhoff Hall",
        "lat": 34.0713, "lng": -118.4440,
        "description": "Study and community space for UCLA student veterans.",
        "features": ["wifi", "comfortable_seating"],
        "noise_level": "quiet",
        "food_drink": False,
        "access": "UCLA student veterans",
        "hours_note": "Check the VRC for current hours.",
        "url": "https://www.asucla.ucla.edu/study-lounges",
        "best_times": ["any time during operating hours"],
        "avoid_times": ["VRC community events"],
    },
]

# ---------------------------------------------------------------------------
# Cafes
# ASUCLA site is JS-rendered — not scrapable with plain requests.
# Data is static, sourced manually from asucla.ucla.edu and dining.ucla.edu.
# Hours change each term — always verify at asucla.ucla.edu/hours.
# ---------------------------------------------------------------------------

ON_CAMPUS_CAFES = [
    {
        "name": "Kerckhoff Coffeehouse",
        "building": "Kerckhoff Hall, 2nd Floor",
        "lat": 34.0713, "lng": -118.4440,
        "description": (
            "On-campus coffeehouse opened in 1975. Serves Sum One coffee, "
            "pastries, sandwiches, and grilled cheese. Indoor seating with outlets "
            "along the interior pillars. Outdoor patio accessible from the main entrance."
        ),
        "coffee": True,
        "food": True,
        "wifi": True,
        "outlets": True,
        "noise_level": "moderate",
        "indoor_outdoor": "both",
        "hours": "Mon-Thu 7am-10pm, Fri 7am-7pm, Sat 8am-6pm, Sun 8am-8pm",
        "hours_source": "asucla.ucla.edu/hours",
        "features": ["wifi", "outlets", "coffee", "food", "outdoor_patio"],
        "url": "https://www.asucla.ucla.edu/ucla/kerckhoff-coffeehouse",
        "best_times": ["weekday mornings before 9:30am", "weekday evenings after 7pm"],
        "avoid_times": ["11am-2pm on weekdays"],
    },
    {
        "name": "Cafe 451",
        "building": "Young Research Library (YRL), 1st Floor",
        "lat": 34.0752, "lng": -118.4418,
        "description": (
            "Peet's cafe on the first floor of YRL. "
            "Serves coffee, pastries, sandwiches, salads, and sushi."
        ),
        "coffee": True,
        "food": True,
        "wifi": True,
        "outlets": True,
        "noise_level": "moderate",
        "indoor_outdoor": "indoor",
        "hours": "Mon-Fri 8am-6pm (varies by term)",
        "hours_source": "library.ucla.edu",
        "features": ["wifi", "outlets", "coffee", "food", "library_adjacent"],
        "url": "https://www.library.ucla.edu/visit/locations/research-library/",
        "best_times": ["weekday mornings", "after 3pm"],
        "avoid_times": ["noon-1:30pm on weekdays"],
    },
    {
        "name": "Jimmy's Coffee House",
        "building": "Lu Valle Commons",
        "lat": 34.0738, "lng": -118.4408,
        "description": (
            "Named after Olympian Jimmy Lu Valle. Serves Peet's coffee, espresso, "
            "matcha, pastries, sushi, and light lunch. Indoor and outdoor seating available."
        ),
        "coffee": True,
        "food": True,
        "wifi": True,
        "outlets": True,
        "noise_level": "low_moderate",
        "indoor_outdoor": "both",
        "hours": "Mon-Thu 7am-9pm, Fri 7am-5pm, Sat 8am-3pm, Sun 10am-4pm",
        "hours_source": "asucla.ucla.edu/hours",
        "features": ["wifi", "outlets", "coffee", "food", "outdoor_seating"],
        "url": "https://www.asucla.ucla.edu/locations",
        "best_times": ["weekday mornings", "mid-afternoon after 2pm"],
        "avoid_times": ["11:30am-1pm on weekdays"],
    },
    {
        "name": "The Study at Hedrick",
        "building": "Hedrick Hall, The Hill",
        "lat": 34.0731, "lng": -118.4484,
        "description": (
            "Cafe with dedicated study areas including private study carrels, "
            "quiet reading rooms, discussion rooms with writable walls, "
            "a fireplace seating area, and natural light from large windows."
        ),
        "coffee": True,
        "food": True,
        "wifi": True,
        "outlets": True,
        "noise_level": "quiet",
        "indoor_outdoor": "indoor",
        "hours": "Check dining.ucla.edu for current hours",
        "hours_source": "dining.ucla.edu",
        "features": [
            "wifi", "outlets", "coffee", "food", "quiet",
            "private_carrels", "group_rooms", "writable_walls", "natural_light",
        ],
        "url": "https://dining.ucla.edu/the-study-at-hedrick/",
        "best_times": ["any time during operating hours"],
        "avoid_times": ["dinner hours when Hill residents pass through"],
    },
    {
        "name": "Music Cafe",
        "building": "Ostin Music Center",
        "lat": 34.0706, "lng": -118.4396,
        "description": (
            "Cafe in the Ostin Music Center near the Inverted Fountain. "
            "Serves Sum One coffee, pastries, and lunch items."
        ),
        "coffee": True,
        "food": True,
        "wifi": True,
        "outlets": True,
        "noise_level": "low_moderate",
        "indoor_outdoor": "indoor",
        "hours": "Mon-Thu 7am-6pm, Fri 7am-5pm, Sat-Sun closed",
        "hours_source": "asucla.ucla.edu/hours",
        "features": ["wifi", "outlets", "coffee", "food"],
        "url": "https://www.asucla.ucla.edu/locations",
        "best_times": ["weekday mornings", "late afternoon before closing"],
        "avoid_times": ["noon-1pm on weekdays"],
    },
    {
        "name": "Bruin Cafe",
        "building": "Sproul Hall",
        "lat": 34.0728, "lng": -118.4476,
        "description": (
            "Dining cafe with indoor seating and an outdoor patio. "
            "Menu includes sandwiches, salads, soups, coffee, tea, and smoothies."
        ),
        "coffee": True,
        "food": True,
        "wifi": True,
        "outlets": True,
        "noise_level": "moderate",
        "indoor_outdoor": "both",
        "hours": "Check dining.ucla.edu for current hours",
        "hours_source": "dining.ucla.edu",
        "features": ["wifi", "outlets", "coffee", "food", "outdoor_patio"],
        "url": "https://dining.ucla.edu/bruin-cafe/",
        "best_times": ["mid-morning and late afternoon on weekdays"],
        "avoid_times": ["meal rush periods"],
    },
    {
        "name": "Anderson School Cafe (Starbucks)",
        "building": "Anderson School of Management",
        "lat": 34.0730, "lng": -118.4397,
        "description": (
            "Starbucks inside Anderson School of Management. "
            "Accepts ASUCLA meal vouchers. "
            "Limited indoor seating; shaded outdoor seating nearby."
        ),
        "coffee": True,
        "food": True,
        "wifi": True,
        "outlets": False,
        "noise_level": "moderate",
        "indoor_outdoor": "both",
        "hours": "Mon-Fri 7am-5pm approx, Sat-Sun closed",
        "hours_source": "asucla.ucla.edu/hours",
        "features": ["wifi", "coffee", "food", "meal_vouchers"],
        "url": "https://www.asucla.ucla.edu/locations",
        "best_times": ["early morning before Anderson classes start"],
        "avoid_times": ["MBA class breaks mid-morning and mid-afternoon"],
    },
    {
        "name": "North Campus Student Center",
        "building": "North Campus Student Center",
        "lat": 34.0753, "lng": -118.4404,
        "description": (
            "Cluster of student-run food options near Rolfe Hall. "
            "Includes coffee, boba, bowls, and quick bites. "
            "Large indoor and outdoor seating areas."
        ),
        "coffee": True,
        "food": True,
        "wifi": True,
        "outlets": True,
        "noise_level": "moderate",
        "indoor_outdoor": "both",
        "hours": "Varies by vendor — generally Mon-Fri daytime",
        "hours_source": "asucla.ucla.edu/hours",
        "features": ["wifi", "outlets", "coffee", "food", "outdoor_seating", "group_friendly"],
        "url": "https://www.asucla.ucla.edu/locations",
        "best_times": ["weekday mornings and mid-afternoon"],
        "avoid_times": ["noon-1pm on weekdays"],
    },
]

# ---------------------------------------------------------------------------
# Outdoor spots
# No authoritative scrapable source — all data is static/curated.
# ---------------------------------------------------------------------------

OUTDOOR_SPOTS = [
    {
        "name": "Franklin D. Murphy Sculpture Garden",
        "area": "North Campus",
        "lat": 34.0754, "lng": -118.4396,
        "description": (
            "Five-acre outdoor sculpture garden with works by Calder, Rodin, Noguchi, "
            "Matisse, and Miro. Benches, shaded tables, and grass areas. "
            "Jacaranda trees bloom in spring."
        ),
        "features": ["outdoor", "benches", "tables", "shaded", "picnic_friendly"],
        "noise_level": "quiet",
        "food_drink": False,
        "food_nearby": "Cafe 451 (YRL) and North Campus Student Center",
        "weather_dependent": True,
        "best_times": ["weekday mornings", "late afternoons"],
        "avoid_times": ["midday in summer"],
    },
    {
        "name": "Kerckhoff Patio",
        "area": "Central Campus",
        "lat": 34.0713, "lng": -118.4438,
        "description": (
            "Outdoor seating directly outside Kerckhoff Hall. "
            "Umbrella-shaded tables and open sunny spots. "
            "Kerckhoff Coffeehouse is accessible from inside the building."
        ),
        "features": ["outdoor", "shaded_tables", "coffee_adjacent"],
        "noise_level": "low_moderate",
        "food_drink": False,
        "food_nearby": "Kerckhoff Coffeehouse",
        "weather_dependent": True,
        "best_times": ["weekday mornings before 10am", "late afternoon after 4pm"],
        "avoid_times": ["midday when shade coverage is lowest"],
    },
    {
        "name": "Inverted Fountain Area",
        "area": "South Campus",
        "lat": 34.0695, "lng": -118.4419,
        "description": (
            "Seating area surrounding the Inverted Fountain (1968). "
            "Tall pine trees and benches on the shaded side. "
            "Water sound reduces ambient noise from nearby walkways."
        ),
        "features": ["outdoor", "shaded", "benches"],
        "noise_level": "quiet",
        "food_drink": False,
        "food_nearby": "Music Cafe (Ostin Music Center)",
        "weather_dependent": True,
        "best_times": ["between class periods on weekdays", "early mornings"],
        "avoid_times": ["passing periods"],
    },
    {
        "name": "Lu Valle Commons Outdoor Area",
        "area": "North Campus",
        "lat": 34.0738, "lng": -118.4408,
        "description": (
            "Outdoor tables and lawn behind Dodd Hall adjacent to Lu Valle Commons. "
            "The courtyard between Dodd Hall and the law school has low foot traffic."
        ),
        "features": ["outdoor", "tables", "shaded", "low_traffic"],
        "noise_level": "quiet",
        "food_drink": False,
        "food_nearby": "Jimmy's Coffee House at Lu Valle Commons",
        "weather_dependent": True,
        "best_times": ["any weekday"],
        "avoid_times": [],
    },
    {
        "name": "Ackerman Union Outdoor Terrace",
        "area": "Central Campus",
        "lat": 34.0707, "lng": -118.4443,
        "description": (
            "Outdoor seating on the Ackerman Union terrace overlooking Ackerman Square. "
            "Shaded tables. Close to printing and food options inside the building."
        ),
        "features": ["outdoor", "shaded", "central_location", "printing_nearby"],
        "noise_level": "moderate",
        "food_drink": False,
        "food_nearby": "Multiple options inside Ackerman Union",
        "weather_dependent": True,
        "best_times": ["mornings before 10am", "late afternoon after 4pm"],
        "avoid_times": ["noon-2pm on weekdays"],
    },
    {
        "name": "Bunche Hall Palm Court",
        "area": "North Campus",
        "lat": 34.0748, "lng": -118.4409,
        "description": (
            "Internal atrium inside Bunche Hall with palm trees spanning four open floors. "
            "Sheltered from wind. Low foot traffic outside of class transition periods."
        ),
        "features": ["sheltered_atrium", "quiet", "natural_light"],
        "noise_level": "quiet",
        "food_drink": False,
        "food_nearby": "North Campus Student Center",
        "weather_dependent": False,
        "best_times": ["mid-morning and afternoon on weekdays"],
        "avoid_times": ["Bunche Hall class transition periods"],
    },
    {
        "name": "Mathias Botanical Garden",
        "area": "Southeast Campus",
        "lat": 34.0666, "lng": -118.4404,
        "description": (
            "Seven-acre botanical garden with plants from around the world. "
            "A stream, tropical specimens, and large trees along walking paths. "
            "Free admission. No outlets or tables — suited to reading physical materials."
        ),
        "features": ["outdoor", "nature", "walking_paths", "free_admission"],
        "noise_level": "very_quiet",
        "food_drink": False,
        "food_nearby": "Limited — Biomedical Library area is closest",
        "weather_dependent": True,
        "best_times": ["warm weekday afternoons"],
        "avoid_times": [],
    },
    {
        "name": "Court of Sciences",
        "area": "South Campus",
        "lat": 34.0685, "lng": -118.4428,
        "description": (
            "Open plaza surrounded by engineering and science buildings. "
            "Benches and tables. Adjacent to Boelter Hall, Young Hall, and the SEL."
        ),
        "features": ["outdoor", "tables", "benches"],
        "noise_level": "low_moderate",
        "food_drink": False,
        "food_nearby": "Bruin Cafe within walking distance",
        "weather_dependent": True,
        "best_times": ["early morning", "late afternoon after 4pm"],
        "avoid_times": ["class transition periods", "midday in summer"],
    },
]

# ---------------------------------------------------------------------------
# Feature tagging
# ---------------------------------------------------------------------------

def _tag_features(text: str) -> List[str]:
    tags = {
        "wifi":           ["wifi", "wi-fi", "wireless"],
        "printing":       ["print", "printer", "wepa"],
        "group_rooms":    ["group study", "study room", "collaboration room"],
        "quiet":          ["quiet", "silent", "reading room"],
        "24_hours":       ["24 hour", "24-hour", "overnight"],
        "outlets":        ["outlet", "power", "charging"],
        "computers":      ["computer", "laptop access", "clicc"],
        "laptop_lending": ["laptop lending"],
        "reservable":     ["reserve", "reservation"],
        "food_nearby":    ["cafe", "coffee", "coffeehouse"],
        "whiteboards":    ["whiteboard"],
        "ada_accessible": ["ada", "accessible", "wheelchair"],
        "natural_light":  ["natural light", "windows"],
    }
    return [tag for tag, kws in tags.items() if any(kw in text for kw in kws)]


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]

# ---------------------------------------------------------------------------
# HTTP fetcher
# ---------------------------------------------------------------------------

class Fetcher:
    def __init__(self, delay: float = DELAY):
        self.delay   = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last   = 0.0
        self._robots: Dict[str, RobotFileParser] = {}

    def _allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            rp = RobotFileParser()
            rp.set_url(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass
            self._robots[origin] = rp
        return self._robots[origin].can_fetch(HEADERS["User-Agent"], url)

    def get(self, url: str) -> Optional[BeautifulSoup]:
        if not self._allowed(url):
            logger.warning("robots.txt disallows: %s", url)
            return None
        elapsed = time.time() - self._last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last = time.time()
        try:
            r = self.session.get(url, timeout=12)
            r.raise_for_status()
            logger.info("GET %s -> %d", url, r.status_code)
            return BeautifulSoup(r.content, "html.parser")
        except Exception as e:
            logger.error("Failed GET %s: %s", url, e)
            return None

# ---------------------------------------------------------------------------
# Library scraper
# ---------------------------------------------------------------------------

def scrape_library_page(fetcher: Fetcher, meta: dict) -> dict:
    url  = meta["url"]
    soup = fetcher.get(url)

    record: dict = {
        "name":         meta["name"],
        "category":     "library",
        "source":       "ucla_library_website",
        "url":          url,
        "lat":          meta["lat"],
        "lng":          meta["lng"],
        "best_times":   meta.get("best_times", []),
        "avoid_times":  meta.get("avoid_times", []),
        "times_note":   TIMES_DISCLAIMER,
        "address":      None,
        "phone":        None,
        "email":        None,
        "description":  None,
        "amenities":    [],
        "features":     [],
        "hours_url":    "https://calendar.library.ucla.edu/hours",
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }

    if not soup:
        record["scrape_error"] = "Could not fetch page"
        return record

    # Description — first substantial paragraph
    for p in (soup.select_one("main") or soup).select("p"):
        text = p.get_text(strip=True)
        if len(text) > 40:
            record["description"] = text[:400]
            break

    # Contact details
    a = soup.select_one("a[href*='map.ucla.edu']")
    if a:
        record["address"] = a.get_text(strip=True)
    ph = soup.select_one("a[href^='tel:']")
    if ph:
        record["phone"] = ph.get_text(strip=True)
    em = soup.select_one("a[href^='mailto:']")
    if em:
        record["email"] = em.get_text(strip=True)

    # Amenities
    amenities: List[str] = []
    for h in soup.find_all(["h2", "h3"]):
        if "at this location" in h.get_text(strip=True).lower():
            sib = h.find_next_sibling()
            while sib and sib.name in ["ul", "p", "div"]:
                for li in sib.find_all("li"):
                    t = li.get_text(strip=True)
                    if t:
                        amenities.append(t)
                sib = sib.find_next_sibling()
            break
    record["amenities"] = amenities

    # Features derived from amenity text + description
    combined = " ".join(amenities).lower() + " " + (record["description"] or "").lower()
    record["features"] = _tag_features(combined)

    return record

# ---------------------------------------------------------------------------
# Collection pipeline
# ---------------------------------------------------------------------------

def collect_all() -> List[dict]:
    fetcher   = Fetcher(delay=DELAY)
    all_spots: List[dict] = []

    print(f"\n{'='*58}")
    print("  UCLA Study Spot Collector")
    print(f"{'='*58}\n")

    # Libraries — scraped from library.ucla.edu
    print(f"[1/4] Scraping {len(LIBRARY_PAGES)} UCLA Library pages...\n")
    for i, meta in enumerate(LIBRARY_PAGES, 1):
        print(f"  [{i:02d}/{len(LIBRARY_PAGES)}] {meta['name']}")
        record = scrape_library_page(fetcher, meta)
        all_spots.append(record)
        (LIBRARY_DIR / f"{_slugify(meta['name'])}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"         {len(record['amenities'])} amenities | {len(record['features'])} features")

    # Lounges — static (ASUCLA is JS-rendered)
    print(f"\n[2/4] Loading {len(ASUCLA_LOUNGES)} ASUCLA study lounges...\n")
    for lounge in ASUCLA_LOUNGES:
        record = {
            **lounge,
            "category":     "study_lounge",
            "source":       "asucla_website_static",
            "times_note":   TIMES_DISCLAIMER,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        all_spots.append(record)
        (LOUNGE_DIR / f"{_slugify(lounge['name'])}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  + {lounge['name']}")

    # Cafes — static (ASUCLA is JS-rendered)
    print(f"\n[3/4] Loading {len(ON_CAMPUS_CAFES)} on-campus cafes...\n")
    for cafe in ON_CAMPUS_CAFES:
        record = {
            **cafe,
            "category":     "cafe",
            "source":       "asucla_dining_static",
            "times_note":   TIMES_DISCLAIMER,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        all_spots.append(record)
        (CAFE_DIR / f"{_slugify(cafe['name'])}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  + {cafe['name']}")

    # Outdoor — static
    print(f"\n[4/4] Loading {len(OUTDOOR_SPOTS)} outdoor spots...\n")
    for spot in OUTDOOR_SPOTS:
        record = {
            **spot,
            "category":     "outdoor",
            "source":       "curated_static",
            "times_note":   TIMES_DISCLAIMER,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        all_spots.append(record)
        (OUTDOOR_DIR / f"{_slugify(spot['name'])}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  + {spot['name']}")

    # Combined output
    combined = DATA_DIR / "all_spots.json"
    combined.write_text(
        json.dumps(all_spots, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved %d spots -> %s", len(all_spots), combined)

    cats: Dict[str, int] = {}
    for s in all_spots:
        cats[s["category"]] = cats.get(s["category"], 0) + 1

    print(f"\n{'='*58}")
    print(f"  Done — {len(all_spots)} spots collected.")
    for cat, n in cats.items():
        print(f"    {cat:20s} {n} spots")
    print(f"\n  Output -> data/raw/ucla/")
    print(f"  Log    -> logs/ucla_collector.log")
    print(f"{'='*58}\n")
    return all_spots


if __name__ == "__main__":
    collect_all()