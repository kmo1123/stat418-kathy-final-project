"""
ucla_study_spots_collector.py
------------------------------
Collects UCLA on-campus study spot data from official UCLA websites
and curated research. Covers four categories:

  1. Libraries     — 8 branch library pages scraped from library.ucla.edu
  2. Study Lounges — 6 ASUCLA student union lounges (asucla.ucla.edu)
  3. Cafes         — 8 on-campus cafes known for study-friendly seating
  4. Outdoor/Other — 8 outdoor spaces, hidden gems, and unique spots

Outputs:
  data/raw/ucla/libraries/     — one JSON per library
  data/raw/ucla/lounges/       — one JSON per lounge
  data/raw/ucla/cafes/         — one JSON per cafe
  data/raw/ucla/outdoor/       — one JSON per outdoor spot
  data/raw/ucla/all_spots.json — combined master file

Usage:
  python ucla_study_spots_collector.py
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

BASE_DIR     = Path(__file__).resolve().parent
LOG_DIR      = BASE_DIR / "logs"
DATA_DIR     = BASE_DIR / "data" / "raw" / "ucla"
LIBRARY_DIR  = DATA_DIR / "libraries"
LOUNGE_DIR   = DATA_DIR / "lounges"
CAFE_DIR     = DATA_DIR / "cafes"
OUTDOOR_DIR  = DATA_DIR / "outdoor"

for d in [LOG_DIR, LIBRARY_DIR, LOUNGE_DIR, CAFE_DIR, OUTDOOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "ucla_collector.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DELAY = 2.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

LIBRARY_PAGES = [
    {"name": "Powell Library", "url": "https://www.library.ucla.edu/visit/locations/powell-library/", "lat": 34.0713, "lng": -118.4417, "insider_note": "Main undergrad library. Night Powell opens overnight during finals week. The Rotunda and Rose Gilbert Reading Room are quietest."},
    {"name": "Charles E. Young Research Library (YRL)", "url": "https://www.library.ucla.edu/visit/locations/research-library/", "lat": 34.0752, "lng": -118.4418, "insider_note": "Graduate-focused. Cafe 451 on 1st floor (named after Fahrenheit 451, written by Bradbury on this campus). Research Commons has high-tech collaborative spaces."},
    {"name": "Science and Engineering Library (SEL)", "url": "https://www.library.ucla.edu/visit/locations/science-engineering-library/", "lat": 34.0692, "lng": -118.4432, "insider_note": "Two locations: Boelter Hall 8th floor and Geology Building. The silent section at the far end of Boelter is legendary. Less crowded than Powell or YRL."},
    {"name": "Louise M. Darling Biomedical Library", "url": "https://www.library.ucla.edu/visit/locations/biomed/", "lat": 34.0663, "lng": -118.4446, "insider_note": "24-hour reading room for health/life sciences grad students. 8 group study rooms on the 6th floor. Less foot traffic than north campus libraries."},
    {"name": "Walter H. Rubsamen Music Library", "url": "https://www.library.ucla.edu/visit/locations/music-library/", "lat": 34.0704, "lng": -118.4398, "insider_note": "Very small and very quiet. Listening stations available. Rarely crowded — a hidden gem for focused solo study."},
    {"name": "Richard C. Rudolph East Asian Library", "url": "https://www.library.ucla.edu/visit/locations/east-asian-library/", "lat": 34.0752, "lng": -118.4415, "insider_note": "Tucked inside YRL. Very quiet. Good option when main YRL floors are packed."},
    {"name": "Management Library", "url": "https://www.library.ucla.edu/visit/locations/management-library/", "lat": 34.0730, "lng": -118.4397, "insider_note": "Located in Cornell Hall at Anderson School of Management. Anderson Starbucks is nearby."},
    {"name": "Arts Library", "url": "https://www.library.ucla.edu/visit/locations/arts-library/", "lat": 34.0736, "lng": -118.4393, "insider_note": "Serves visual arts, architecture, and urban planning. Good natural light. Less crowded than big undergraduate libraries."},
]

ASUCLA_LOUNGES = [
    {"name": "Louise Kerckhoff Study Lounge", "building": "Kerckhoff Hall, 3rd Floor", "lat": 34.0713, "lng": -118.4440, "description": "Originally the Women's Smoking Room, now a co-ed study lounge with comfortable couches, bright lighting, and stained glass bay windows.", "features": ["comfortable_seating", "quiet", "wifi", "historic_space", "outlets"], "noise_level": "quiet", "food_drink": False, "access": "UCLA students, faculty, staff", "hours_note": "Check ASUCLA for current hours.", "url": "https://www.asucla.ucla.edu/study-lounges", "insider_note": "One of the most beautiful study spots on campus. Get there early."},
    {"name": "Kerckhoff Second Floor Study Lounge", "building": "Kerckhoff Hall, 2nd Floor", "lat": 34.0713, "lng": -118.4440, "description": "Smaller, cozier version of the 3rd floor lounge. Steps from Kerckhoff Coffeehouse.", "features": ["comfortable_seating", "quiet", "wifi", "coffee_nearby"], "noise_level": "quiet", "food_drink": False, "access": "UCLA students, faculty, staff", "hours_note": "Check ASUCLA for current hours.", "url": "https://www.asucla.ucla.edu/study-lounges", "insider_note": "Less known than the 3rd floor lounge — often easier to get a seat."},
    {"name": "Ackerman Union Open Study", "building": "Ackerman Union", "lat": 34.0707, "lng": -118.4440, "description": "Any unreserved meeting room in Ackerman Union is available as open study space. Check the schedule posted next to each room entrance.", "features": ["wifi", "flexible_seating", "group_friendly", "printing_nearby"], "noise_level": "moderate", "food_drink": False, "access": "UCLA students, faculty, staff (unreserved rooms only)", "hours_note": "Varies by room.", "url": "https://www.asucla.ucla.edu/study-lounges", "insider_note": "Hit or miss depending on room availability. Good for groups."},
    {"name": "Bruin Reflection Space", "building": "Ackerman Union, 3rd Floor", "lat": 34.0707, "lng": -118.4440, "description": "Non-denominational space for contemplation, meditation, and prayer. Open to all Bruins. Extremely quiet.", "features": ["quiet", "wifi", "calm_atmosphere", "comfortable_seating"], "noise_level": "silent", "food_drink": False, "access": "All Bruins", "hours_note": "Check ASUCLA for current hours.", "url": "https://www.asucla.ucla.edu/study-lounges", "insider_note": "Underrated for solo focused work. Very few people know about it."},
    {"name": "Transfer Student Center Lounge", "building": "Ackerman Union", "lat": 34.0707, "lng": -118.4440, "description": "Lounge inside the UCLA Transfer Student Center. Also supports commuter and non-traditional students.", "features": ["wifi", "comfortable_seating", "support_services_nearby"], "noise_level": "moderate", "food_drink": False, "access": "All students (transfer-focused)", "hours_note": "Check transfer center hours.", "url": "https://www.asucla.ucla.edu/study-lounges", "insider_note": "Welcoming and rarely crowded. Good for commuters between classes."},
    {"name": "Veteran Resource Center", "building": "Kerckhoff Hall", "lat": 34.0713, "lng": -118.4440, "description": "Welcoming space for UCLA student veterans. Open for studying and community.", "features": ["wifi", "comfortable_seating", "community_focused"], "noise_level": "quiet", "food_drink": False, "access": "UCLA student veterans", "hours_note": "Check VRC for current hours.", "url": "https://www.asucla.ucla.edu/study-lounges", "insider_note": "Tight-knit community. Great for veteran students."},
]

ON_CAMPUS_CAFES = [
    {"name": "Kerckhoff Coffeehouse", "building": "Kerckhoff Hall, 2nd Floor", "lat": 34.0713, "lng": -118.4440, "description": "UCLA's first coffee house (est. 1975). Stunning gothic architecture. Sum One coffee, pastries, sandwiches, grilled cheese + soup. Tables with outlets along the pillars.", "coffee": True, "food": True, "wifi": True, "outlets": True, "outlets_note": "Outlets along the pillars — grab one early.", "noise_level": "moderate", "indoor_outdoor": "both", "hours": "Mon-Thu 7am-10pm, Fri 7am-7pm, Sat 8am-6pm, Sun 8am-8pm", "features": ["wifi", "outlets", "coffee", "food", "historic_building", "outdoor_patio"], "insider_note": "Iconic spot but tables fill fast. The outdoor patio is less crowded and great in good weather.", "url": "https://www.asucla.ucla.edu/ucla/kerckhoff-coffeehouse"},
    {"name": "Cafe 451", "building": "Young Research Library (YRL), 1st Floor", "lat": 34.0752, "lng": -118.4418, "description": "Named after Fahrenheit 451, written by Bradbury on this campus. Peet's coffee, pastries, sandwiches, salads, and sushi. Cozy study spot inside YRL.", "coffee": True, "food": True, "wifi": True, "outlets": True, "outlets_note": "Library wifi and outlets throughout YRL.", "noise_level": "moderate", "indoor_outdoor": "indoor", "hours": "Check YRL hours — typically Mon-Fri 8am-6pm", "features": ["wifi", "outlets", "coffee", "food", "library_adjacent", "quiet_nearby"], "insider_note": "Best combo: coffee here then head upstairs to the quiet YRL stacks.", "url": "https://www.asucla.ucla.edu/locations"},
    {"name": "Jimmy's Coffee House", "building": "Lu Valle Commons", "lat": 34.0738, "lng": -118.4408, "description": "Named after Olympian Jimmy Lu Valle. Peet's coffee, espresso, matcha, pastries, sushi, light lunch. Indoor seating almost always less crowded than Kerckhoff.", "coffee": True, "food": True, "wifi": True, "outlets": True, "outlets_note": "Some outlets — try to snag a table near the wall.", "noise_level": "low_moderate", "indoor_outdoor": "both", "hours": "Mon-Thu 7am-9pm, Fri 7am-5pm, Sat 8am-3pm, Sun 10am-4pm", "features": ["wifi", "outlets", "coffee", "food", "less_crowded", "outdoor_seating"], "insider_note": "Student favorite — less hectic than Kerckhoff. The outdoor tables behind Dodd Hall nearby are a hidden gem.", "url": "https://www.asucla.ucla.edu/locations"},
    {"name": "The Study at Hedrick", "building": "Hedrick Hall, The Hill", "lat": 34.0731, "lng": -118.4484, "description": "European-style bakery cafe that doubles as a serious study space. Private study carrels, quiet reading rooms, discussion rooms with writable walls, fireplace, natural light, and abundant outlets.", "coffee": True, "food": True, "wifi": True, "outlets": True, "outlets_note": "Abundant outlets throughout.", "noise_level": "quiet", "indoor_outdoor": "indoor", "hours": "Check dining.ucla.edu for current hours", "features": ["wifi", "outlets", "coffee", "food", "quiet", "private_carrels", "group_rooms", "fireplace", "natural_light", "writable_walls"], "insider_note": "Arguably the best study cafe on campus. The noise-canceling fish bowl carrels are legendary.", "url": "https://dining.ucla.edu/the-study-at-hedrick/"},
    {"name": "Music Cafe", "building": "Ostin Music Center", "lat": 34.0706, "lng": -118.4396, "description": "Modern cafe in the renovated Ostin Music Center near the Inverted Fountain. Sum One coffee, pastries, lunch items. Great atmosphere for music lovers and artists.", "coffee": True, "food": True, "wifi": True, "outlets": True, "outlets_note": "A few outlets — arrive early for a good seat.", "noise_level": "low_moderate", "indoor_outdoor": "indoor", "hours": "Mon-Thu 7am-6pm, Fri 7am-5pm, Sat-Sun Closed", "features": ["wifi", "outlets", "coffee", "food", "artistic_atmosphere"], "insider_note": "Less known than Kerckhoff but equally charming. Next to the Inverted Fountain — a calming spot.", "url": "https://www.asucla.ucla.edu/locations"},
    {"name": "Bruin Cafe", "building": "Sproul Hall", "lat": 34.0728, "lng": -118.4476, "description": "Casual dining cafe with indoor space and outdoor patio. Sandwiches, salads, soups, specialty coffees, teas, smoothies. Both indoor and outdoor areas are great study spots.", "coffee": True, "food": True, "wifi": True, "outlets": True, "outlets_note": "Outlets available indoors.", "noise_level": "moderate", "indoor_outdoor": "both", "hours": "Check dining.ucla.edu for current hours", "features": ["wifi", "outlets", "coffee", "food", "outdoor_patio", "on_the_hill"], "insider_note": "Convenient for Hill residents. Outdoor patio is great on sunny LA days.", "url": "https://dining.ucla.edu/bruin-cafe/"},
    {"name": "Anderson School Cafe (Starbucks)", "building": "Anderson School of Management", "lat": 34.0730, "lng": -118.4397, "description": "Starbucks inside Anderson School of Management. Accepts ASUCLA meal vouchers. Shaded outdoor seating nearby.", "coffee": True, "food": True, "wifi": True, "outlets": False, "outlets_note": "Limited outlets — mostly grab-and-go.", "noise_level": "moderate", "indoor_outdoor": "both", "hours": "Mon-Fri 7am-5pm approx, Sat-Sun Closed", "features": ["wifi", "coffee", "food", "shaded_outdoor_seating", "meal_vouchers"], "insider_note": "Great for a quick coffee before heading to the Management Library next door.", "url": "https://www.asucla.ucla.edu/locations"},
    {"name": "North Campus Student Center", "building": "North Campus Student Center", "lat": 34.0753, "lng": -118.4404, "description": "Cluster of student-run restaurants near Rolfe Hall and the Robert Graham Sculpture Court. Includes coffee, boba, bowls, and quick bites. Large indoor and outdoor seating — popular study hub.", "coffee": True, "food": True, "wifi": True, "outlets": True, "outlets_note": "Outlets in seating areas — availability varies.", "noise_level": "moderate", "indoor_outdoor": "both", "hours": "Varies — generally Mon-Fri daytime", "features": ["wifi", "outlets", "coffee", "food", "outdoor_seating", "group_friendly"], "insider_note": "Rapid access to coffee, boba, and food makes this a great base for long study days.", "url": "https://www.asucla.ucla.edu/locations"},
]

OUTDOOR_SPOTS = [
    {"name": "Franklin D. Murphy Sculpture Garden", "area": "North Campus", "lat": 34.0754, "lng": -118.4396, "description": "5-acre outdoor sculpture museum with works by Calder, Rodin, Noguchi, Matisse, and Miro. Grassy hills, benches, shaded tables among sculptures and flowering trees. Jacaranda trees bloom brilliantly in spring.", "features": ["outdoor", "peaceful", "benches", "tables", "shaded", "picnic_friendly"], "noise_level": "quiet", "food_drink": False, "food_nearby": "Cafe 451 (YRL) and North Campus Student Center nearby", "best_for": "solo study, reading, mental reset", "weather_dependent": True, "insider_note": "One of the most beautiful spots on campus. Especially magical during jacaranda season (April-May)."},
    {"name": "Kerckhoff Patio", "area": "Central Campus", "lat": 34.0713, "lng": -118.4438, "description": "Outdoor patio directly outside Kerckhoff Hall. Tables shaded by umbrellas or trees plus open sunny spots. Steps from Kerckhoff Coffeehouse. Indoor/outdoor combo unbeatable on good weather days.", "features": ["outdoor", "shaded_tables", "sunny_spots", "coffee_steps_away"], "noise_level": "low_moderate", "food_drink": False, "food_nearby": "Kerckhoff Coffeehouse steps away", "best_for": "casual study, outdoor coffee sessions", "weather_dependent": True, "insider_note": "Go early — tables fill fast. The shaded umbrella tables are premium spots."},
    {"name": "Inverted Fountain Area", "area": "South Campus (near Franz Hall)", "lat": 34.0695, "lng": -118.4419, "description": "The iconic Inverted Fountain (1968) where water flows downward. Surrounded by tall pine trees and benches. The sound of flowing water creates a naturally calming study environment.", "features": ["outdoor", "shaded", "benches", "calming_sound", "iconic"], "noise_level": "quiet", "food_drink": False, "food_nearby": "Music Cafe (Ostin Music Center) adjacent", "best_for": "reading, solo study, mental reset", "weather_dependent": True, "insider_note": "Underrated as a study spot. Most people walk past rather than stop. The shaded pine benches are extremely peaceful."},
    {"name": "Lu Valle Commons Outdoor Area", "area": "North Campus (behind Dodd Hall)", "lat": 34.0738, "lng": -118.4408, "description": "The private lawn and outdoor tables behind Dodd Hall next to Lu Valle Commons. The nearby courtyard between Dodd Hall and the law school is secluded and peaceful.", "features": ["outdoor", "tables", "shaded", "private_feel", "coffee_adjacent"], "noise_level": "quiet", "food_drink": False, "food_nearby": "Jimmy's Coffee House at Lu Valle Commons", "best_for": "solo or pair study, reading between North Campus classes", "weather_dependent": True, "insider_note": "A genuinely hidden gem. The Dodd/law school courtyard is very secluded. Perfect for students with North Campus classes."},
    {"name": "Ackerman Union Outdoor Terrace", "area": "Central Campus", "lat": 34.0707, "lng": -118.4443, "description": "Outdoor seating area on the Ackerman Union terrace overlooking Ackerman Square. Shaded seating with a view of central campus. Close to printing, food, and the UCLA Store.", "features": ["outdoor", "shaded", "central_location", "views", "printing_nearby"], "noise_level": "moderate", "food_drink": False, "food_nearby": "Multiple ASUCLA options inside Ackerman Union", "best_for": "casual study, group meetups, between-class sessions", "weather_dependent": True, "insider_note": "Gets busy mid-day. Better in the morning or late afternoon."},
    {"name": "Bunche Hall Palm Court", "area": "North Campus", "lat": 34.0748, "lng": -118.4409, "description": "A hidden atrium inside Bunche Hall (social sciences building). Palm tree-filled internal courtyard spanning four floors. Quiet, sheltered from wind, largely unknown outside social science students.", "features": ["outdoor_feel", "sheltered", "quiet", "hidden_gem", "natural_light"], "noise_level": "quiet", "food_drink": False, "food_nearby": "North Campus Student Center nearby", "best_for": "solo focus work, reading", "weather_dependent": False, "insider_note": "Most people walk right past it. Feels outdoors but protected from weather. One of campus's true hidden gems."},
    {"name": "Mathias Botanical Garden", "area": "Southeast Campus", "lat": 34.0666, "lng": -118.4404, "description": "7-acre botanical garden with plant specimens from around the world. A stream, tropical plants, towering old trees, and flowers. Free admission. Best for reading or unwinding during long study days.", "features": ["outdoor", "nature", "peaceful", "free_admission", "walking_paths"], "noise_level": "very_quiet", "food_drink": False, "food_nearby": "Limited — Biomedical Library area", "best_for": "reading, decompression, light study", "weather_dependent": True, "insider_note": "Not for laptop work — no outlets. Perfect for reading a textbook on a warm afternoon or a mental break during finals."},
    {"name": "Court of Sciences", "area": "South Campus", "lat": 34.0685, "lng": -118.4428, "description": "Open plaza surrounded by engineering and science buildings. Benches, tables, and open space used by STEM students between classes. Easy access to Boelter Hall, Young Hall, and the SEL.", "features": ["outdoor", "tables", "benches", "collaborative", "stem_adjacent"], "noise_level": "low_moderate", "food_drink": False, "food_nearby": "Vending machines nearby; Bruin Cafe within walking distance", "best_for": "group study, collaborative work, between-class sessions for STEM students", "weather_dependent": True, "insider_note": "Best during off-peak hours. Limited shade — bring sunscreen for long sessions."},
]


class Fetcher:
    def __init__(self, delay=DELAY):
        self.delay   = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last   = 0.0
        self._robots = {}

    def _allowed(self, url):
        from urllib.parse import urlparse
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

    def get(self, url):
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


def _tag_features(text):
    tags = {
        "wifi": ["wifi", "wi-fi", "wireless"],
        "printing": ["print", "printer", "wepa"],
        "group_rooms": ["group study", "study room", "collaboration"],
        "quiet": ["quiet", "silent", "reading room"],
        "24_hours": ["24 hour", "24-hour", "overnight", "night powell"],
        "outlets": ["outlet", "power", "charging"],
        "computers": ["computer", "laptop", "clicc"],
        "laptop_lending": ["laptop lending"],
        "reservable": ["reserve", "reservation"],
        "food_nearby": ["cafe", "coffee", "coffeehouse"],
        "whiteboards": ["whiteboard"],
        "3d_printing": ["3d print", "makerbot", "lux lab"],
        "ada_accessible": ["ada", "accessible", "wheelchair"],
        "outdoor": ["outdoor", "patio", "garden"],
        "natural_light": ["natural light", "windows"],
    }
    return [tag for tag, kws in tags.items() if any(kw in text for kw in kws)]


def _slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def scrape_library_page(fetcher, meta):
    url  = meta["url"]
    soup = fetcher.get(url)
    record = {
        "name": meta["name"], "category": "library", "source": "ucla_library_website",
        "url": url, "lat": meta["lat"], "lng": meta["lng"],
        "insider_note": meta.get("insider_note"),
        "address": None, "phone": None, "email": None, "description": None,
        "amenities": [], "spaces": [], "features": [],
        "hours_url": "https://calendar.library.ucla.edu/hours",
        "reservations_url": None,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    if not soup:
        record["scrape_error"] = "Could not fetch page"
        return record

    # Description
    for p in (soup.select_one("main") or soup).select("p"):
        text = p.get_text(strip=True)
        if len(text) > 40:
            record["description"] = text[:400]
            break

    # Address / phone / email
    a = soup.select_one("a[href*='map.ucla.edu']")
    if a: record["address"] = a.get_text(strip=True)
    p = soup.select_one("a[href^='tel:']")
    if p: record["phone"] = p.get_text(strip=True)
    e = soup.select_one("a[href^='mailto:']")
    if e: record["email"] = e.get_text(strip=True)

    # Amenities
    amenities = []
    for h in soup.find_all(["h2", "h3"]):
        if "at this location" in h.get_text(strip=True).lower():
            sib = h.find_next_sibling()
            while sib and sib.name in ["ul", "p", "div"]:
                for li in sib.find_all("li"):
                    t = li.get_text(strip=True)
                    if t: amenities.append(t)
                sib = sib.find_next_sibling()
            break
    record["amenities"] = amenities

    # Spaces
    spaces = []
    in_spaces = False
    for el in soup.find_all(["h2", "h3"]):
        if el.name == "h2" and el.get_text(strip=True).lower() == "spaces":
            in_spaces = True
            continue
        if in_spaces:
            if el.name == "h2": break
            text = el.get_text(strip=True)
            if text:
                np = el.find_next_sibling("p")
                spaces.append({
                    "space_name": text,
                    "description": np.get_text(strip=True)[:200] if np else None,
                    "reservable": bool(el.find_next("a", string=re.compile(r"reserve", re.I)))
                })
    record["spaces"] = spaces

    rl = soup.select_one("a[href*='calendar.library.ucla.edu']")
    if rl: record["reservations_url"] = rl["href"]

    all_text = " ".join(amenities + [s["space_name"] for s in spaces]).lower()
    all_text += " " + (record["description"] or "").lower()
    record["features"] = _tag_features(all_text)
    return record


def collect_all():
    fetcher   = Fetcher(delay=DELAY)
    all_spots = []

    print(f"\n{'='*58}")
    print(f"  UCLA Study Spot Collector")
    print(f"{'='*58}\n")

    print(f"[1/4] Scraping {len(LIBRARY_PAGES)} UCLA Library branch pages...\n")
    for i, meta in enumerate(LIBRARY_PAGES, 1):
        print(f"  [{i:02d}/{len(LIBRARY_PAGES)}] {meta['name']}")
        record = scrape_library_page(fetcher, meta)
        all_spots.append(record)
        (LIBRARY_DIR / f"{_slugify(meta['name'])}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"         {len(record['amenities'])} amenities | {len(record['spaces'])} spaces | {len(record['features'])} features")

    print(f"\n[2/4] Loading {len(ASUCLA_LOUNGES)} ASUCLA study lounges...\n")
    for lounge in ASUCLA_LOUNGES:
        record = {**lounge, "category": "study_lounge", "source": "asucla_website",
                  "collected_at": datetime.now(timezone.utc).isoformat()}
        all_spots.append(record)
        (LOUNGE_DIR / f"{_slugify(lounge['name'])}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  + {lounge['name']}")

    print(f"\n[3/4] Loading {len(ON_CAMPUS_CAFES)} on-campus study cafes...\n")
    for cafe in ON_CAMPUS_CAFES:
        record = {**cafe, "category": "cafe", "source": "curated_asucla_dining",
                  "collected_at": datetime.now(timezone.utc).isoformat()}
        all_spots.append(record)
        (CAFE_DIR / f"{_slugify(cafe['name'])}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  + {cafe['name']}")

    print(f"\n[4/4] Loading {len(OUTDOOR_SPOTS)} outdoor spots and hidden gems...\n")
    for spot in OUTDOOR_SPOTS:
        record = {**spot, "category": "outdoor", "source": "curated",
                  "collected_at": datetime.now(timezone.utc).isoformat()}
        all_spots.append(record)
        (OUTDOOR_DIR / f"{_slugify(spot['name'])}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  + {spot['name']}")

    combined = DATA_DIR / "all_spots.json"
    combined.write_text(json.dumps(all_spots, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d spots -> %s", len(all_spots), combined)

    cats = {}
    for s in all_spots:
        cats[s["category"]] = cats.get(s["category"], 0) + 1

    print(f"\n{'='*58}")
    print(f"  Done! {len(all_spots)} spots collected.")
    for cat, n in cats.items():
        print(f"    {cat:20s} {n} spots")
    print(f"\n  Libraries -> data/raw/ucla/libraries/")
    print(f"  Lounges   -> data/raw/ucla/lounges/")
    print(f"  Cafes     -> data/raw/ucla/cafes/")
    print(f"  Outdoor   -> data/raw/ucla/outdoor/")
    print(f"  Combined  -> data/raw/ucla/all_spots.json")
    print(f"  Log       -> logs/ucla_collector.log")
    print(f"{'='*58}\n")
    return all_spots


if __name__ == "__main__":
    collect_all()