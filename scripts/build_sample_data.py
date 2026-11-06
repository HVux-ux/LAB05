#!/usr/bin/env python3
"""Build the bundled Vietnam travel sample dataset in three formats.

Why this script exists
----------------------
The lab asks for practice with the CSV, JSON and JSONL formats. Rather than
hand-maintaining three files that silently drift apart, this script holds a
single source of truth in Python and *renders* it three times:

    data/sample/travel_vn.csv    flat, one table, list columns joined with "|"
    data/sample/travel_vn.json   pretty-printed array, list columns as arrays
    data/sample/travel_vn.jsonl  one JSON object per line (JSON Lines)

The rendered files are committed so the app runs without any build step, but
they are always reproducible:

    python scripts/build_sample_data.py

The sample is intentionally small and hand-curated. `scripts/download_kaggle.py`
fetches a real Kaggle dataset, and `src/travel_planner/adapter.py` maps its
columns onto this canonical schema.
"""

from __future__ import annotations

import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sample"

# --------------------------------------------------------------------------
# Canonical fields. Every row must carry all of them; empty values are "".
# --------------------------------------------------------------------------
FIELDS: list[str] = [
    "id",
    "name",
    "city",
    "region",
    "category",
    "subcategory",
    "description",
    "tags",
    "price_vnd",
    "price_usd",
    "rating",
    "reviews",
    "duration_hours",
    "best_months",
    "address",
    "latitude",
    "longitude",
    "source",
]

CATEGORIES = {"landmark", "food", "accommodation", "experience", "transport"}


def item(
    id: str,
    name: str,
    city: str,
    region: str,
    category: str,
    subcategory: str,
    description: str,
    tags: list[str],
    price_vnd: int,
    price_usd: float,
    rating: float,
    reviews: int,
    duration_hours: float,
    best_months: list[int],
    address: str,
    latitude: float,
    longitude: float,
) -> dict:
    assert category in CATEGORIES, f"bad category {category!r} for {id}"
    return {
        "id": id,
        "name": name,
        "city": city,
        "region": region,
        "category": category,
        "subcategory": subcategory,
        "description": description,
        "tags": tags,
        "price_vnd": price_vnd,
        "price_usd": price_usd,
        "rating": rating,
        "reviews": reviews,
        "duration_hours": duration_hours,
        "best_months": best_months,
        "address": address,
        "latitude": latitude,
        "longitude": longitude,
        "source": "sample",
    }


# --------------------------------------------------------------------------
# Regions: north / central / south. Used by the app's region filter.
# --------------------------------------------------------------------------
ITEMS: list[dict] = [
    # ----------------------------- HANOI -----------------------------
    item("HN-LM-01", "Hoan Kiem Lake and Ngoc Son Temple", "Hanoi", "north", "landmark",
         "lake", "The historic heart of the Old Quarter: a 15th-century temple on a small island, "
         "connected by the red Huc Bridge. Busiest at dawn when locals practise tai chi.",
         ["walkable", "cheap", "photography", "history"], 30000, 1.2, 4.6, 21430,
         1.5, [1, 2, 3, 4, 9, 10, 11, 12], "Dinh Tien Hoang St, Hoan Kiem, Hanoi", 21.0287, 105.8524),
    item("HN-LM-02", "Temple of Literature", "Hanoi", "north", "landmark",
         "temple", "Vietnam's first university (1070), built as a Confucian temple. Five courtyards "
         "of stelae recording royal exam graduates.",
         ["history", "architecture", "culture", "quiet"], 30000, 1.2, 4.7, 15220,
         1.5, [1, 2, 3, 4, 5, 10, 11, 12], "58 Quoc Tu Giam St, Dong Da, Hanoi", 21.0272, 105.8356),
    item("HN-LM-03", "Hanoi Old Quarter walking loop", "Hanoi", "north", "experience",
         "walking tour", "A 36-street grid where each street historically sold one trade. Ideal as a "
         "self-guided evening walk with food stops.",
         ["walkable", "street food", "nightlife", "photography"], 0, 0.0, 4.8, 30110,
         3.0, [1, 2, 3, 4, 5, 9, 10, 11, 12], "Old Quarter, Hoan Kiem, Hanoi", 21.0345, 105.8500),
    item("HN-LM-04", "Ho Chi Minh Mausoleum and Ba Dinh Square", "Hanoi", "north", "landmark",
         "monument", "Granite mausoleum on the square where independence was declared in 1945. "
         "Opens only in the morning and closes several weeks per year for maintenance.",
         ["history", "architecture", "free"], 0, 0.0, 4.5, 12980,
         2.0, [1, 2, 3, 4, 11, 12], "19 Ngoc Ha St, Ba Dinh, Hanoi", 21.0368, 105.8347),
    item("HN-FD-01", "Bun Cha Huong Lien", "Hanoi", "north", "food",
         "street food", "Grilled pork with rice noodles and a sweet-sour dipping broth: the Hanoi dish "
         "most associated with a famous 2016 state dinner.",
         ["street food", "local favourite", "lunch", "cheap"], 60000, 2.4, 4.6, 8420,
         1.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "24 Le Van Huu, Hai Ba Trung, Hanoi", 21.0148, 105.8509),
    item("HN-FD-02", "Cafe Giang egg coffee", "Hanoi", "north", "food",
         "cafe", "Whipped egg yolk over strong coffee, invented in Hanoi in 1946. The alley entrance "
         "makes it feel like a hidden find.",
         ["coffee", "dessert", "rainy day", "cheap"], 45000, 1.8, 4.5, 9600,
         0.8, [1, 2, 3, 4, 5, 9, 10, 11, 12], "39 Nguyen Huu Huan, Hoan Kiem, Hanoi", 21.0329, 105.8524),
    item("HN-AC-01", "Old Quarter boutique garden hotel", "Hanoi", "north", "accommodation",
         "hotel", "Mid-range hotel with a rooftop terrace walking distance from the night market and "
         "Hoan Kiem Lake.",
         ["central", "rooftop", "breakfast", "mid-range"], 1450000, 58.0, 4.5, 3180,
         0.0, [1, 2, 3, 4, 5, 9, 10, 11, 12], "Hang Be st, Hoan Kiem, Hanoi", 21.0357, 105.8492),
    item("HN-AC-02", "Lakeside five-star heritage hotel", "Hanoi", "north", "accommodation",
         "hotel", "Colonial-era grande dame facing the opera house, with the city's best-known "
         "high tea and a large outdoor pool.",
         ["luxury", "central", "pool", "spa"], 5400000, 216.0, 4.8, 5410,
         0.0, [1, 2, 3, 4, 5, 10, 11, 12], "1 Le Thanh Tong, Hoan Kiem, Hanoi", 21.0245, 105.8574),
    item("HN-TR-01", "Noi Bai Airport to Old Quarter by private car", "Hanoi", "north", "transport",
         "airport transfer", "Pre-booked sedan, roughly 45 minutes into the centre, tolls included.",
         ["airport", "door to door", "fixed price"], 420000, 17.0, 4.4, 2210,
         0.9, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Noi Bai International Airport, Hanoi", 21.2187, 105.8042),

    # ---------------------------- NINH BINH ---------------------------
    item("NB-LM-01", "Trang An boat route", "Ninh Binh", "north", "experience",
         "boat trip", "Two-hour rowboat through limestone karsts and flooded caves. The landscape "
         "that gave the region its nickname, Ha Long Bay on land.",
         ["nature", "boat", "photography", "unesco"], 250000, 10.0, 4.8, 11240,
         2.5, [3, 4, 5, 9, 10, 11], "Trang An, Hoa Lu, Ninh Binh", 20.2480, 105.8985),
    item("NB-LM-02", "Hang Mua viewpoint stairs", "Ninh Binh", "north", "landmark",
         "viewpoint", "Around 500 stone steps up to a dragon-shaped summit with the best panorama "
         "over the Tam Coc river valley.",
         ["nature", "hiking", "photography", "sunrise"], 100000, 4.0, 4.7, 8830,
         1.5, [3, 4, 5, 9, 10, 11], "Hang Mua, Hoa Lu, Ninh Binh", 20.2277, 105.9247),
    item("NB-AC-01", "Karst-view homestay in Tam Coc", "Ninh Binh", "north", "accommodation",
         "homestay", "Family homestay with bungalows facing the limestone cliffs; bicycles and "
         "breakfast are included.",
         ["nature", "quiet", "budget", "family"], 750000, 30.0, 4.6, 1980,
         0.0, [3, 4, 5, 9, 10, 11], "Hai Nham, Hoa Lu, Ninh Binh", 20.2180, 105.9290),
    item("NB-LM-03", "Tam Coc riverside cycle loop", "Ninh Binh", "north", "experience",
         "cycling", "Flat 12 km loop between rice fields, karst towers and small villages; bicycles "
         "are rented at the homestays.",
         ["nature", "cycling", "cheap", "family"], 80000, 3.2, 4.5, 3120,
         2.0, [3, 4, 5, 6, 9, 10, 11], "Tam Coc, Hoa Lu, Ninh Binh", 20.2160, 105.9320),

    # ------------------------------ SAPA ------------------------------
    item("SP-LM-01", "Fansipan summit cable car", "Sa Pa", "north", "experience",
         "cable car", "Cable car to 3,143 m, the highest point in Indochina. Clear mornings in the "
         "cool season give views over the Hoang Lien range.",
         ["mountain", "views", "family", "cold"], 850000, 34.0, 4.3, 6410,
         4.0, [3, 4, 9, 10, 11], "Fansipan Cable Car Station, Sa Pa", 22.3350, 103.8400),
    item("SP-LM-02", "Muong Hoa valley trek to Lao Chai", "Sa Pa", "north", "experience",
         "trekking", "Half-day guided descent through terraced rice fields to Hmong and Dao villages, "
         "usually with lunch at a local family's house.",
         ["mountain", "trekking", "culture", "photography"], 550000, 22.0, 4.7, 4820,
         6.0, [4, 5, 9, 10, 11], "Muong Hoa Valley, Sa Pa", 22.3250, 103.8600),
    item("SP-AC-01", "Cloud-view mountain lodge", "Sa Pa", "north", "accommodation",
         "lodge", "Wooden lodge above the valley with a fireplace and a glass-walled breakfast room.",
         ["mountain", "quiet", "mid-range", "views"], 1800000, 72.0, 4.6, 2140,
         0.0, [3, 4, 9, 10, 11], "Ta Van, Sa Pa", 22.3180, 103.8900),

    # ---------------------------- HA LONG -----------------------------
    item("HL-LM-01", "Ha Long Bay overnight cruise", "Ha Long", "north", "experience",
         "cruise", "One night on a junk among 1,600 limestone islands, with kayaking, a cave visit "
         "and dinner on deck.",
         ["unesco", "boat", "nature", "luxury"], 3200000, 128.0, 4.6, 9310,
         24.0, [3, 4, 5, 9, 10, 11], "Tuan Chau Marina, Ha Long", 20.9100, 106.9900),
    item("HL-AC-01", "Marina-side hotel with bay view", "Ha Long", "north", "accommodation",
         "hotel", "Tower hotel above the marina, with a pool deck facing the islands and a "
         "shuttle to the cruise piers.",
         ["bay view", "pool", "mid-range", "family"], 1600000, 64.0, 4.4, 2410,
         0.0, [3, 4, 5, 9, 10, 11], "Hoang Quoc Viet St, Ha Long", 20.9550, 107.0700),
    item("HL-LM-02", "Bai Tu Long day cruise", "Ha Long", "north", "experience",
         "cruise", "A quieter day route east of the main bay, with fewer boats and a fishing village stop.",
         ["unesco", "boat", "nature", "quiet"], 1450000, 58.0, 4.4, 3110,
         9.0, [3, 4, 5, 9, 10, 11], "Hon Gai Harbour, Ha Long", 20.9500, 107.0800),

    # ------------------------------ HUE -------------------------------
    item("HU-LM-01", "Imperial Citadel and Forbidden Purple City", "Hue", "central", "landmark",
         "citadel", "Walled 1804 imperial city with moat, gates and restored pavilions; several "
         "sections are still being rebuilt after the war.",
         ["history", "architecture", "unesco", "culture"], 200000, 8.0, 4.6, 14210,
         3.0, [1, 2, 3, 4, 5, 9, 10, 11, 12], "23 Thang 8 St, Hue", 16.4698, 107.5790),
    item("HU-LM-02", "Thien Mu Pagoda", "Hue", "central", "landmark",
         "pagoda", "Seven-storey riverside pagoda from 1601, the tallest in Vietnam, and a working "
         "monastery.",
         ["history", "riverside", "quiet", "free"], 0, 0.0, 4.5, 6120,
         1.0, [1, 2, 3, 4, 5, 9, 10, 11, 12], "Huong Long, Hue", 16.4533, 107.5450),
    item("HU-FD-01", "Bun bo Hue at Dong Ba market", "Hue", "central", "food",
         "street food", "Spicy beef noodle soup with lemongrass and shrimp paste, eaten at the "
         "market from early morning.",
         ["street food", "breakfast", "spicy", "cheap"], 45000, 1.8, 4.7, 5210,
         0.8, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Dong Ba Market, Hue", 16.4712, 107.5840),
    item("HU-AC-01", "Perfume River garden guesthouse", "Hue", "central", "accommodation",
         "guesthouse", "Small garden guesthouse on the north bank, ten minutes by foot from the citadel.",
         ["riverside", "quiet", "budget", "garden"], 780000, 31.0, 4.4, 1620,
         0.0, [1, 2, 3, 4, 5, 9, 10, 11, 12], "Nguyen Cong Tru St, Hue", 16.4740, 107.5870),

    # ----------------------------- DA NANG ----------------------------
    item("DN-LM-01", "Golden Bridge and Ba Na Hills", "Da Nang", "central", "landmark",
         "theme park", "Giant stone hands holding a curved golden walkway above the clouds, reached "
         "by one of the world's longest cable cars.",
         ["photography", "family", "views", "touristy"], 900000, 36.0, 4.2, 19340,
         5.0, [2, 3, 4, 5, 6, 7, 8, 9], "Hoa Ninh, Hoa Vang, Da Nang", 15.9980, 107.9880),
    item("DN-LM-02", "Marble Mountains", "Da Nang", "central", "landmark",
         "caves", "Five marble hills with cave shrines, tunnels and a lift-accessible summit "
         "viewpoint over the coast.",
         ["history", "caves", "views", "half day"], 40000, 1.6, 4.4, 9870,
         2.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Hoa Hai, Ngu Hanh Son, Da Nang", 16.0036, 108.2630),
    item("DN-LM-03", "My Khe beach day", "Da Nang", "central", "experience",
         "beach", "Long city beach with lifeguards, cafes and the best swimming window between "
         "April and August.",
         ["beach", "swimming", "free", "family"], 0, 0.0, 4.5, 7210,
         3.0, [4, 5, 6, 7, 8, 9], "Vo Nguyen Giap St, Da Nang", 16.0544, 108.2478),
    item("DN-FD-01", "Seafood dinner on the Han River", "Da Nang", "central", "food",
         "seafood", "Grilled scallops, clams and fish hotpot at riverside tables with a view of the "
         "Dragon Bridge.",
         ["seafood", "dinner", "riverside", "mid-range"], 450000, 18.0, 4.5, 4310,
         1.5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Bach Dang St, Hai Chau, Da Nang", 16.0730, 108.2240),
    item("DN-AC-01", "Beachfront hotel with infinity pool", "Da Nang", "central", "accommodation",
         "hotel", "High-rise hotel across the road from My Khe, with a rooftop infinity pool and "
         "direct beach access.",
         ["beach", "pool", "views", "mid-range"], 2100000, 84.0, 4.6, 6120,
         0.0, [2, 3, 4, 5, 6, 7, 8, 9], "Nguyen Van Linh, Da Nang", 16.0600, 108.2380),

    # ----------------------------- HOI AN -----------------------------
    item("HA-LM-01", "Hoi An Ancient Town at lantern hour", "Hoi An", "central", "landmark",
         "old town", "Preserved trading port of yellow merchant houses; after dark the streets are "
         "lit only by silk lanterns and the river is full of floating candles.",
         ["unesco", "photography", "evening", "walkable"], 120000, 4.8, 4.8, 24110,
         2.5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Chau Thuong Van, Hoi An", 15.8777, 108.3274),
    item("HA-LM-02", "Tra Que vegetable village cooking class", "Hoi An", "central", "experience",
         "cooking class", "Cycle to an organic herb village, then cook four central-Vietnamese dishes "
         "with a local family.",
         ["food", "hands-on", "family", "half day"], 700000, 28.0, 4.8, 5420,
         4.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Tra Que, Cam Ha, Hoi An", 15.8930, 108.3360),
    item("HA-LM-03", "An Bang beach", "Hoi An", "central", "experience",
         "beach", "Four kilometres of sand with beach clubs at one end and empty stretches at the "
         "other; good sunset swimming.",
         ["beach", "swimming", "sunset", "free"], 0, 0.0, 4.5, 3410,
         3.0, [4, 5, 6, 7, 8, 9], "An Bang, Cam An, Hoi An", 15.9150, 108.3420),
    item("HA-FD-01", "Cao lau stall in the old town", "Hoi An", "central", "food",
         "noodle soup", "Thick rice noodles, braised pork and crunchy greens: the dish is made only "
         "with water from one local well.",
         ["local favourite", "lunch", "cheap", "must try"], 50000, 2.0, 4.6, 4120,
         0.8, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Tran Phu St, Hoi An", 15.8770, 108.3280),
    item("HA-AC-01", "Riverside boutique in the old town", "Hoi An", "central", "accommodation",
         "hotel", "Small boutique hotel inside the pedestrian zone, with bicycles included and a "
         "plunge pool in the courtyard.",
         ["central", "pool", "quiet", "mid-range"], 1650000, 66.0, 4.7, 2890,
         0.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Nguyen Thai Hoc St, Hoi An", 15.8800, 108.3260),

    # --------------------------- NHA TRANG ----------------------------
    item("NT-LM-01", "Four Island boat trip", "Nha Trang", "central", "experience",
         "boat trip", "Day boat around Hon Mun and Hon Mot with snorkelling stops and a floating "
         "seafood lunch.",
         ["beach", "snorkelling", "boat", "family"], 550000, 22.0, 4.2, 5210,
         7.0, [1, 2, 3, 4, 5, 6, 7, 8], "Cau Da Port, Nha Trang", 12.2010, 109.2110),
    item("NT-LM-02", "Po Nagar Cham Towers", "Nha Trang", "central", "landmark",
         "temple", "Brick towers built between the 7th and 12th centuries by the Cham kingdom, still "
         "used for worship.",
         ["history", "architecture", "culture", "cheap"], 30000, 1.2, 4.3, 4310,
         1.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "2 Thang 4 St, Nha Trang", 12.2650, 109.1950),
    item("NT-LM-03", "Mud bath and hot spring", "Nha Trang", "central", "experience",
         "spa", "Mineral mud baths and hot pools on the edge of town; a common rainy-afternoon plan.",
         ["wellness", "rainy day", "family", "relaxing"], 400000, 16.0, 4.4, 3710,
         2.0, [1, 2, 3, 4, 5, 9, 10, 11, 12], "Nguyen Tat Thanh Blvd, Nha Trang", 12.2760, 109.1720),
    item("NT-AC-01", "Seafront hotel on Tran Phu", "Nha Trang", "central", "accommodation",
         "hotel", "Tower hotel with sea-view rooms, a small pool and a promenade address.",
         ["beach", "pool", "views", "budget"], 1150000, 46.0, 4.3, 5120,
         0.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Tran Phu St, Nha Trang", 12.2380, 109.1960),

    # ----------------------------- DA LAT -----------------------------
    item("DL-LM-01", "Crazy House", "Da Lat", "central", "landmark",
         "architecture", "Surreal guesthouse of winding concrete staircases and organic shapes, "
         "designed by architect Dang Viet Nga.",
         ["architecture", "quirky", "photography", "family"], 80000, 3.2, 4.1, 8120,
         1.5, [1, 2, 3, 4, 5, 9, 10, 11, 12], "3 Huynh Thuc Khang, Da Lat", 11.9350, 108.4330),
    item("DL-LM-02", "Coffee farm and roastery visit", "Da Lat", "central", "experience",
         "coffee tour", "Da Lat grows most of Vietnam's arabica; a farm walk covers picking, honey "
         "processing and cupping.",
         ["coffee", "nature", "learning", "half day"], 350000, 14.0, 4.7, 2410,
         3.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Trai Mat, Da Lat", 11.9800, 108.4900),
    item("DL-LM-03", "Tuyen Lam Lake kayaking", "Da Lat", "central", "experience",
         "kayaking", "Pine-ringed lake twenty minutes from town, calm enough for beginners, with a "
         "cafe at the jetty.",
         ["nature", "kayaking", "quiet", "family"], 200000, 8.0, 4.4, 1620,
         2.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Tuyen Lam Lake, Da Lat", 11.8900, 108.4300),
    item("DL-AC-01", "Pine hill homestay", "Da Lat", "central", "accommodation",
         "homestay", "Family homestay with a wood stove, a big shared kitchen and flower gardens "
         "below the town centre.",
         ["mountain", "quiet", "budget", "garden"], 620000, 25.0, 4.5, 1420,
         0.0, [1, 2, 3, 4, 5, 9, 10, 11, 12], "Mai Anh Dao St, Da Lat", 11.9420, 108.4470),

    # ---------------------------- HO CHI MINH -------------------------
    item("SG-LM-01", "War Remnants Museum", "Ho Chi Minh City", "south", "landmark",
         "museum", "Documentary museum on the war years, with photographic archives and captured "
         "aircraft. Emotionally heavy; expect 2 to 3 hours.",
         ["history", "museum", "indoor", "rainy day"], 40000, 1.6, 4.6, 21430,
         2.5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "28 Vo Van Tan, District 3, HCMC", 10.7798, 106.6920),
    item("SG-LM-02", "Central Post Office and Notre Dame", "Ho Chi Minh City", "south", "landmark",
         "architecture", "Two French-era landmarks facing each other on Dong Khoi; the post office is "
         "still operating.",
         ["architecture", "history", "photography", "free"], 0, 0.0, 4.4, 17320,
         1.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "2 Cong xa Paris, District 1, HCMC", 10.7801, 106.6995),
    item("SG-LM-03", "Ben Thanh market food crawl", "Ho Chi Minh City", "south", "experience",
         "food tour", "Self-guided evening circuit of the market and the surrounding alleys: banh mi, "
         "broken rice, sugarcane juice and grilled skewers.",
         ["street food", "evening", "walkable", "cheap"], 250000, 10.0, 4.5, 9120,
         2.5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Le Loi St, District 1, HCMC", 10.7725, 106.6980),
    item("SG-LM-04", "Cu Chi tunnels half day", "Ho Chi Minh City", "south", "experience",
         "historical tour", "Guided visit to the tunnel network used during the war, with a food-stop "
         "on the drive out. Two hours each way.",
         ["history", "day trip", "learning", "half day"], 400000, 16.0, 4.3, 12410,
         6.0, [1, 2, 3, 4, 5, 11, 12], "Cu Chi District, HCMC", 10.9650, 106.4940),
    item("SG-FD-01", "Modern Vietnamese tasting menu", "Ho Chi Minh City", "south", "food",
         "fine dining", "Eight-course menu reinterpreting southern street food, with optional "
         "local-wine pairing. Book a week ahead.",
         ["fine dining", "dinner", "romantic", "splurge"], 2600000, 104.0, 4.7, 3410,
         2.5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "District 1, HCMC", 10.7750, 106.7050),
    item("SG-AC-01", "Rooftop-bar hotel in District 1", "Ho Chi Minh City", "south", "accommodation",
         "hotel", "Central hotel five minutes from the walking street, with a rooftop bar and gym.",
         ["central", "nightlife", "mid-range", "gym"], 1900000, 76.0, 4.5, 7310,
         0.0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Ngo Van Nam St, District 1, HCMC", 10.7780, 106.7060),
    item("SG-TR-01", "Tan Son Nhat airport transfer", "Ho Chi Minh City", "south", "transport",
         "airport transfer", "Fixed-fare car from the airport to District 1, roughly 25 minutes "
         "outside rush hour.",
         ["airport", "door to door", "fixed price"], 250000, 10.0, 4.2, 1810,
         0.6, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Tan Son Nhat Airport, HCMC", 10.8188, 106.6520),

    # --------------------------- MEKONG DELTA -------------------------
    item("CT-LM-01", "Cai Rang floating market", "Can Tho", "south", "experience",
         "boat trip", "Dawn boat market where wholesale produce is sold from sampans; the earliest "
         "boats arrive around 5 am.",
         ["nature", "boat", "early morning", "photography"], 300000, 12.0, 4.5, 4210,
         3.0, [1, 2, 3, 11, 12], "Cai Rang, Can Tho", 10.0000, 105.7500),
    item("CT-LM-02", "Mekong homestay and cooking night", "Can Tho", "south", "accommodation",
         "homestay", "Bungalow stay in a fruit orchard with a cooking evening and bicycle use.",
         ["nature", "food", "budget", "family"], 700000, 28.0, 4.6, 1210,
         0.0, [1, 2, 3, 4, 5, 11, 12], "An Binh, Ninh Kieu, Can Tho", 10.0200, 105.7600),

    # ----------------------------- PHU QUOC ---------------------------
    item("PQ-LM-01", "Sunset cable car to Hon Thom", "Phu Quoc", "south", "experience",
         "cable car", "One of the longest over-sea cable cars, running from the south of the island "
         "to Hon Thom, timed to sunset.",
         ["beach", "views", "family", "sunset"], 650000, 26.0, 4.4, 8210,
         3.0, [1, 2, 3, 4, 11, 12], "An Thoi, Phu Quoc", 10.0070, 104.0100),
    item("PQ-LM-02", "Sao Beach and island snorkel", "Phu Quoc", "south", "experience",
         "beach", "White sand and shallow water on the east side of the island, with snorkel boats "
         "leaving mid-morning.",
         ["beach", "swimming", "snorkelling", "family"], 200000, 8.0, 4.5, 6120,
         4.0, [1, 2, 3, 4, 11, 12], "Bai Sao, An Thoi, Phu Quoc", 10.0450, 104.0330),
    item("PQ-AC-01", "Beach bungalow resort", "Phu Quoc", "south", "accommodation",
         "resort", "Garden bungalows a short walk from the sand, with kayaks, a pool and breakfast "
         "included.",
         ["beach", "pool", "luxury", "quiet"], 3400000, 136.0, 4.7, 4210,
         0.0, [1, 2, 3, 4, 11, 12], "Ong Lang, Phu Quoc", 10.2000, 103.9500),
    item("PQ-FD-01", "Night market seafood grill", "Phu Quoc", "south", "food",
         "seafood", "Pick your fish at the ice counter and have it grilled at the table; scallops and "
         "sea urchin are the local picks.",
         ["seafood", "evening", "mid-range", "local favourite"], 500000, 20.0, 4.4, 3410,
         1.5, [1, 2, 3, 4, 11, 12], "Duong Dong, Phu Quoc", 10.2170, 103.9650),
]

# City metadata: a second, differently shaped JSON document on purpose (an
# object with a "cities" array), so the loader has to handle both shapes.
CITIES: list[dict] = [
    {"city": "Hanoi", "region": "north", "airport": "HAN",
     "best_months": [10, 11, 12, 1, 2, 3, 4], "daily_budget_usd": {"budget": 35, "mid": 75, "luxury": 190}},
    {"city": "Ninh Binh", "region": "north", "airport": "HAN",
     "best_months": [3, 4, 5, 9, 10, 11], "daily_budget_usd": {"budget": 28, "mid": 55, "luxury": 120}},
    {"city": "Sa Pa", "region": "north", "airport": "HAN",
     "best_months": [3, 4, 9, 10, 11], "daily_budget_usd": {"budget": 30, "mid": 62, "luxury": 140}},
    {"city": "Ha Long", "region": "north", "airport": "HAN",
     "best_months": [3, 4, 5, 9, 10, 11], "daily_budget_usd": {"budget": 45, "mid": 95, "luxury": 240}},
    {"city": "Hue", "region": "central", "airport": "HUI",
     "best_months": [1, 2, 3, 4, 5, 9, 10, 11, 12], "daily_budget_usd": {"budget": 30, "mid": 62, "luxury": 150}},
    {"city": "Da Nang", "region": "central", "airport": "DAD",
     "best_months": [2, 3, 4, 5, 6, 7, 8], "daily_budget_usd": {"budget": 32, "mid": 70, "luxury": 175}},
    {"city": "Hoi An", "region": "central", "airport": "DAD",
     "best_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "daily_budget_usd": {"budget": 34, "mid": 74, "luxury": 185}},
    {"city": "Nha Trang", "region": "central", "airport": "CXR",
     "best_months": [1, 2, 3, 4, 5, 6, 7, 8], "daily_budget_usd": {"budget": 30, "mid": 65, "luxury": 160}},
    {"city": "Da Lat", "region": "central", "airport": "DLI",
     "best_months": [1, 2, 3, 4, 5, 9, 10, 11, 12], "daily_budget_usd": {"budget": 28, "mid": 58, "luxury": 135}},
    {"city": "Ho Chi Minh City", "region": "south", "airport": "SGN",
     "best_months": [1, 2, 3, 4, 5, 11, 12], "daily_budget_usd": {"budget": 38, "mid": 82, "luxury": 210}},
    {"city": "Can Tho", "region": "south", "airport": "SGN",
     "best_months": [1, 2, 3, 11, 12], "daily_budget_usd": {"budget": 26, "mid": 52, "luxury": 115}},
    {"city": "Phu Quoc", "region": "south", "airport": "PQC",
     "best_months": [1, 2, 3, 4, 11, 12], "daily_budget_usd": {"budget": 40, "mid": 88, "luxury": 230}},
]


def write_csv(rows: list[dict], path: pathlib.Path) -> None:
    """CSV is flat: list columns are joined into a single delimited cell."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["tags"] = "|".join(row["tags"])
            flat["best_months"] = "|".join(str(m) for m in row["best_months"])
            writer.writerow(flat)


def write_json(rows: list[dict], path: pathlib.Path) -> None:
    """JSON keeps the natural nested shape and stays human-readable."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"schema_version": 1, "items": rows}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_jsonl(rows: list[dict], path: pathlib.Path) -> None:
    """JSONL: one self-contained object per line, no enclosing array."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids = [r["id"] for r in ITEMS]
    assert len(ids) == len(set(ids)), "duplicate item id"
    assert {r["city"] for r in ITEMS} <= {c["city"] for c in CITIES}, "item cites unknown city"

    write_csv(ITEMS, OUT_DIR / "travel_vn.csv")
    write_json(ITEMS, OUT_DIR / "travel_vn.json")
    write_jsonl(ITEMS, OUT_DIR / "travel_vn.jsonl")

    with (OUT_DIR / "cities.json").open("w", encoding="utf-8") as fh:
        json.dump({"schema_version": 1, "cities": CITIES}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"wrote {len(ITEMS)} items and {len(CITIES)} cities to {OUT_DIR}")
    for path in sorted(OUT_DIR.iterdir()):
        print(f"  {path.name:<18} {path.stat().st_size:>6} bytes")


if __name__ == "__main__":
    main()
