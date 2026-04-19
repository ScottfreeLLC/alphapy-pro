"""Generate pizza toppings train/test CSVs for AlphaPy ranking pipeline.

Trend scores for train rows are anchored to real signals from the PMQ Pizza
Power Report 2026, Tastewise 2026 data, and Datassential trend coverage
(hot honey, pistachio, paneer, plant-based pepperoni, Korean BBQ, pickles).
Test rows are emerging or speculative "undiscovered" candidates — their
trend_score is intentionally left blank for prediction.
"""

from __future__ import annotations

import csv
from pathlib import Path

COLS = [
    "name",
    "category",        # group_id for ranking
    "cuisine",
    "flavor_family",
    "sweet", "salt", "umami", "heat", "acid", "bitter", "fat",
    "crunch", "melt", "moist", "heat_tol",
    "buzz", "chef_adopt", "novelty", "insta",
    "years_on_menu", "price_tier",
    "is_fermented", "is_plant_based", "is_premium",
    "trend_score",
]


def row(name, category, cuisine, family,
        sweet, salt, umami, heat, acid, bitter, fat,
        crunch, melt, moist, heat_tol,
        buzz, chef, novelty, insta,
        years, price, fermented, plant, premium, trend):
    return [
        name, category, cuisine, family,
        sweet, salt, umami, heat, acid, bitter, fat,
        crunch, melt, moist, heat_tol,
        buzz, chef, novelty, insta,
        years, price, fermented, plant, premium, trend,
    ]


# ---------------------------------------------------------------------------
# TRAIN: toppings with observed trend signals (2023-2026)
# trend_score is a 0-100 momentum score blending 12-mo growth + chef-adoption
# ---------------------------------------------------------------------------

TRAIN = [
    # name, cat, cuisine, family, sweet,salt,umami,heat,acid,bitter,fat, crunch,melt,moist,heat_tol, buzz,chef,novelty,insta, years,price, ferm,plant,prem, trend

    # --- Classics (low momentum, high base) ---
    row("Pepperoni",            "meat",    "italian_american", "savory",         1, 8, 9, 3, 2, 1, 8,  2, 0, 2, 10,  8, 5, 1, 7, 60, 2, 0, 0, 0, 18),
    row("Italian Sausage",      "meat",    "italian",          "savory",         1, 7, 8, 3, 1, 1, 7,  2, 0, 3, 10,  6, 4, 1, 5, 50, 2, 0, 0, 0, 12),
    row("Button Mushroom",      "produce", "italian",          "umami_bomb",     1, 3, 7, 0, 1, 2, 2,  2, 0, 6,  9,  5, 4, 1, 4, 50, 1, 0, 0, 0, 10),
    row("Green Bell Pepper",    "produce", "italian",          "bright_acidic",  3, 2, 2, 1, 3, 2, 1,  4, 0, 6,  9,  4, 3, 1, 4, 55, 1, 0, 1, 0,  6),
    row("Yellow Onion",         "produce", "italian",          "savory",         3, 2, 5, 0, 2, 1, 1,  4, 0, 5,  9,  4, 4, 1, 3, 60, 1, 0, 1, 0,  7),
    row("Black Olive",          "produce", "mediterranean",    "savory",         0, 8, 5, 0, 3, 3, 5,  2, 0, 4, 10,  4, 4, 1, 4, 50, 1, 0, 1, 0,  6),
    row("Green Olive",          "produce", "mediterranean",    "bright_acidic",  0, 8, 4, 0, 5, 2, 5,  2, 0, 4, 10,  4, 4, 1, 3, 45, 1, 0, 1, 0,  5),
    row("Mozzarella",           "cheese",  "italian",          "savory",         2, 4, 5, 0, 1, 0, 7,  0, 10,7,  9,  8, 6, 1, 6, 60, 2, 0, 0, 0, 14),
    row("Ham",                  "meat",    "italian_american", "savory",         2, 7, 6, 0, 1, 0, 5,  1, 0, 3, 10,  4, 3, 1, 3, 50, 2, 0, 0, 0,  5),
    row("Bacon",                "meat",    "american",         "smoky",          2, 8, 8, 1, 1, 1, 8,  7, 0, 2, 10,  8, 6, 1, 8, 40, 3, 0, 0, 0, 22),
    row("Pineapple",            "fruit",   "hawaiian",         "sweet",          8, 1, 2, 0, 5, 1, 1,  2, 0, 8,  9,  6, 3, 2, 8, 40, 2, 0, 1, 0,  3),
    row("Jalapeno",             "produce", "mexican_american", "spicy",          2, 3, 2, 7, 2, 1, 1,  4, 0, 6,  9,  7, 5, 2, 7, 40, 1, 0, 1, 0, 24),
    row("Anchovy",              "seafood", "italian",          "umami_bomb",     0,10, 9, 0, 4, 2, 6,  2, 0, 5, 10,  4, 3, 1, 3, 60, 3, 0, 0, 0,  8),
    row("Pesto",                "sauce",   "italian",          "herbal",         2, 4, 6, 1, 3, 3, 7,  0, 0, 8,  8,  7, 6, 1, 6, 25, 3, 0, 0, 0, 28),
    row("San Marzano Tomato",   "produce", "italian",          "bright_acidic",  5, 3, 7, 0, 6, 1, 1,  1, 0, 9,  9,  7, 6, 1, 8, 60, 3, 0, 0, 1, 20),
    row("Ricotta",              "cheese",  "italian",          "savory",         3, 3, 4, 0, 2, 0, 6,  0, 4, 7,  8,  7, 6, 1, 5, 40, 2, 0, 0, 0, 32),
    row("Parmigiano Reggiano",  "cheese",  "italian",          "umami_bomb",     1, 8, 9, 0, 2, 1, 6,  3, 2, 2, 10,  8, 7, 1, 8, 60, 4, 0, 0, 1, 18),
    row("Baby Spinach",         "produce", "italian",          "bright_acidic",  1, 1, 3, 0, 2, 4, 1,  2, 0, 6,  8,  5, 4, 1, 5, 40, 1, 0, 1, 0, 12),
    row("Artichoke Heart",      "produce", "mediterranean",    "bright_acidic",  2, 4, 4, 0, 5, 3, 3,  3, 0, 6,  9,  5, 4, 1, 5, 30, 2, 0, 1, 0,  9),
    row("Sun-Dried Tomato",     "produce", "mediterranean",    "umami_bomb",     6, 5, 8, 0, 6, 1, 2,  4, 0, 3,  9,  5, 4, 1, 5, 25, 2, 0, 0, 0, 14),
    row("Roasted Red Pepper",   "produce", "mediterranean",    "sweet",          5, 2, 5, 0, 3, 1, 2,  2, 0, 7,  9,  5, 5, 1, 5, 30, 2, 0, 1, 0, 11),
    row("Feta",                 "cheese",  "mediterranean",    "bright_acidic",  1, 8, 4, 0, 4, 1, 6,  2, 2, 5,  8,  7, 5, 1, 6, 30, 2, 1, 0, 0, 26),
    row("Goat Cheese (Chevre)", "cheese",  "french",           "bright_acidic",  2, 5, 4, 0, 5, 2, 7,  1, 4, 5,  8,  7, 6, 1, 7, 25, 3, 1, 0, 1, 34),
    row("Arugula",              "produce", "italian",          "bright_acidic",  1, 2, 3, 0, 3, 4, 1,  3, 0, 5,  7,  7, 6, 1, 7, 20, 2, 0, 1, 0, 22),
    row("Prosciutto di Parma",  "meat",    "italian",          "umami_bomb",     1, 8, 8, 0, 2, 0, 7,  1, 0, 4,  6,  9, 8, 1, 9, 30, 4, 0, 0, 1, 38),
    row("Salami",               "meat",    "italian",          "savory",         1, 7, 7, 2, 2, 1, 7,  3, 0, 3,  9,  5, 4, 1, 5, 50, 2, 0, 0, 0, 10),
    row("Soppressata",          "meat",    "italian",          "spicy",          1, 7, 7, 4, 2, 1, 7,  3, 0, 3, 10,  7, 5, 1, 7, 20, 3, 0, 0, 1, 32),
    row("Hot Honey",            "sauce",   "american",         "sweet",          9, 2, 2, 6, 3, 0, 2,  0, 0, 8,  7,  9, 8, 3, 9, 4,  2, 0, 0, 0, 95),
    row("Nashville Hot Chicken","meat",    "american_south",   "spicy",          2, 5, 6, 9, 2, 1, 7,  7, 0, 5,  9,  9, 7, 7, 9, 4,  3, 0, 0, 0, 78),
    row("Buffalo Chicken",      "meat",    "american",         "spicy",          1, 6, 5, 6, 5, 1, 6,  5, 0, 6,  9,  7, 6, 2, 7, 15, 2, 0, 0, 0, 40),
    row("Korean Bulgogi Beef",  "meat",    "korean",           "umami_bomb",     6, 6, 7, 2, 3, 1, 6,  3, 0, 5,  9,  9, 8, 7, 9, 6,  3, 0, 0, 1, 72),
    row("Pistachio",            "herb_spice","italian_med",    "savory",         2, 4, 5, 0, 1, 2, 6,  7, 0, 2,  9,  9, 8, 8, 9, 4,  3, 0, 1, 1, 86),
    row("Mortadella",           "meat",    "italian",          "savory",         2, 6, 7, 0, 1, 1, 7,  1, 0, 4,  8,  8, 6, 2, 8, 8,  3, 0, 0, 1, 58),
    row("Burrata",              "cheese",  "italian",          "savory",         2, 4, 5, 0, 1, 0, 8,  0, 5, 9,  5,  9, 7, 1, 9, 10, 4, 0, 0, 1, 65),
    row("Brie",                 "cheese",  "french",           "savory",         3, 4, 5, 0, 2, 1, 7,  0, 8, 8,  7,  7, 6, 1, 7, 10, 3, 1, 0, 1, 42),
    row("Smoked Gouda",         "cheese",  "dutch_american",   "smoky",          2, 5, 7, 0, 1, 1, 7,  1, 8, 4,  9,  7, 6, 2, 7, 12, 3, 0, 0, 0, 48),
    row("Tandoori Paneer",      "cheese",  "indian",           "spicy",          2, 5, 6, 5, 3, 1, 6,  3, 0, 5,  9,  9, 8, 7, 9, 5,  3, 1, 0, 0, 82),
    row("Dill Pickles",         "produce", "american_deli",    "funky_fermented",1, 7, 3, 0, 7, 1, 1,  5, 0, 7,  8,  9, 8, 3, 9, 3,  2, 1, 1, 0, 70),
    row("Pepperoncini",         "produce", "mediterranean",    "bright_acidic",  1, 5, 3, 3, 5, 1, 1,  4, 0, 6,  9,  7, 5, 2, 7, 10, 2, 1, 1, 0, 44),
    row("Kimchi",               "produce", "korean",           "funky_fermented",2, 4, 6, 6, 7, 1, 1,  4, 0, 7,  7,  9, 8, 5, 9, 5,  4, 1, 1, 0, 75),
    row("Sauerkraut",           "produce", "german",           "funky_fermented",1, 5, 4, 0, 7, 1, 1,  4, 0, 7,  8,  5, 4, 2, 6, 6,  2, 1, 1, 0, 40),
    row("Microgreens",          "produce", "modern",           "bright_acidic",  1, 1, 2, 0, 2, 3, 1,  3, 0, 4,  4,  9, 9, 2, 9, 5,  3, 0, 1, 1, 48),
    row("Plant-Based Pepperoni","meat",    "modern_plant",     "savory",         1, 7, 6, 3, 2, 1, 6,  2, 0, 3,  9,  9, 9, 8, 9, 3,  3, 0, 1, 1, 83),
    row("Chili Crisp",          "sauce",   "sichuan",          "spicy",          2, 6, 8, 7, 3, 2, 6,  6, 0, 4,  9,  9, 9, 5, 9, 4,  3, 0, 0, 0, 88),
    row("Porcini Mushroom",     "produce", "italian",          "umami_bomb",     1, 3, 9, 0, 1, 2, 2,  2, 0, 5,  9,  7, 5, 3, 7, 15, 4, 0, 0, 1, 48),
    row("Oyster Mushroom",      "produce", "asian",            "umami_bomb",     1, 2, 7, 0, 1, 2, 2,  3, 0, 5,  9,  7, 6, 3, 7, 10, 3, 0, 0, 0, 52),
    row("Chanterelle",          "produce", "european",         "umami_bomb",     2, 2, 7, 0, 1, 3, 2,  3, 0, 5,  9,  6, 5, 3, 6, 15, 4, 0, 0, 1, 38),
    row("Fig Jam",              "sauce",   "mediterranean",    "sweet",          8, 1, 2, 0, 3, 2, 1,  0, 0, 7,  8,  7, 6, 2, 7, 10, 3, 1, 0, 1, 46),
    row("Caramelized Onion",    "produce", "french",           "sweet",          6, 3, 6, 0, 2, 1, 2,  2, 0, 6,  8,  7, 5, 2, 7, 15, 2, 0, 1, 0, 44),
    row("Truffle Oil",          "sauce",   "french_italian",   "umami_bomb",     1, 2, 9, 0, 1, 3, 6,  0, 0, 5,  6,  7, 6, 3, 8, 12, 5, 0, 0, 1, 35),
    row("Guanciale",            "meat",    "italian",          "savory",         1, 7, 8, 0, 1, 0, 9,  4, 0, 2, 10,  7, 6, 2, 8, 15, 4, 0, 0, 1, 42),
    row("Nduja",                "meat",    "italian_calabrian","spicy",          1, 6, 7, 7, 2, 1, 8,  0, 3, 7,  9,  9, 8, 6, 8, 8,  4, 0, 0, 1, 72),
    row("Calabrian Chili",      "produce", "italian",          "spicy",          2, 4, 4, 7, 3, 2, 4,  2, 0, 6,  9,  8, 7, 5, 8, 8,  3, 1, 1, 0, 62),
    row("Gorgonzola",           "cheese",  "italian",          "pungent",        1, 7, 6, 0, 2, 5, 7,  0, 7, 5,  7,  6, 5, 1, 7, 30, 3, 0, 0, 1, 24),
    row("Smoked Mozzarella",    "cheese",  "italian",          "smoky",          2, 5, 6, 0, 1, 1, 7,  0, 9, 7,  9,  6, 4, 1, 6, 15, 3, 0, 0, 0, 30),
    row("Manchego",             "cheese",  "spanish",          "umami_bomb",     1, 6, 6, 0, 1, 1, 6,  2, 3, 4,  9,  6, 4, 1, 6, 15, 3, 0, 0, 1, 24),
    row("Chorizo",              "meat",    "spanish_mexican",  "spicy",          2, 6, 7, 6, 2, 1, 7,  3, 0, 4,  9,  7, 6, 3, 7, 15, 3, 0, 0, 0, 46),
    row("Thai Basil",           "herb_spice","thai",           "herbal",         1, 1, 2, 1, 2, 3, 1,  2, 0, 6,  6,  7, 6, 3, 7, 8,  3, 0, 1, 0, 38),
    row("Fresh Cilantro",       "herb_spice","mexican_asian",  "herbal",         1, 1, 2, 0, 3, 3, 1,  2, 0, 6,  6,  5, 4, 1, 5, 30, 1, 0, 1, 0, 20),
    row("Lemon Zest",           "herb_spice","mediterranean",  "bright_acidic",  2, 1, 2, 0, 7, 3, 1,  1, 0, 4,  6,  7, 6, 2, 7, 10, 2, 0, 1, 0, 28),
    row("Tomato Confit",        "produce", "french",           "umami_bomb",     5, 2, 7, 0, 5, 1, 4,  0, 0, 8,  8,  7, 6, 2, 7, 10, 3, 0, 1, 1, 34),
    row("Roasted Garlic",       "produce", "mediterranean",    "umami_bomb",     4, 2, 6, 0, 1, 2, 3,  2, 0, 6,  9,  6, 5, 2, 6, 25, 1, 0, 1, 0, 26),
    row("Pickled Red Onion",    "produce", "mexican",          "bright_acidic",  3, 3, 2, 0, 6, 1, 1,  4, 0, 6,  8,  8, 7, 3, 8, 8,  2, 1, 1, 0, 58),
    row("Gochujang Sauce",      "sauce",   "korean",           "spicy",          4, 5, 7, 6, 3, 1, 3,  0, 0, 8,  8,  9, 8, 5, 9, 5,  3, 1, 0, 0, 78),
    row("Tandoori Chicken",     "meat",    "indian",           "spicy",          2, 5, 6, 5, 3, 1, 4,  4, 0, 5,  9,  8, 7, 5, 8, 6,  3, 0, 0, 0, 62),
    row("Butter Chicken",       "meat",    "indian",           "umami_bomb",     4, 5, 7, 3, 3, 1, 7,  2, 0, 8,  8,  9, 8, 7, 9, 5,  3, 0, 0, 1, 84),
    row("Chicken Tikka Masala", "meat",    "indian",           "umami_bomb",     3, 5, 7, 4, 4, 1, 6,  2, 0, 8,  8,  9, 8, 7, 9, 5,  3, 0, 0, 1, 80),
    row("Saag Paneer",          "cheese",  "indian",           "herbal",         2, 4, 6, 2, 2, 3, 5,  1, 0, 7,  8,  8, 7, 6, 8, 5,  3, 1, 0, 0, 66),
    row("Garam Masala",         "herb_spice","indian",         "spicy",          2, 2, 4, 5, 2, 3, 1,  2, 0, 3,  8,  8, 7, 6, 7, 8,  2, 0, 1, 0, 52),
    row("Mango Chutney",        "sauce",   "indian",           "sweet",          8, 2, 2, 3, 5, 1, 1,  0, 0, 7,  8,  7, 6, 4, 7, 6,  2, 1, 1, 0, 48),
    row("Harissa",              "sauce",   "north_african",    "spicy",          2, 4, 5, 7, 4, 1, 4,  0, 0, 7,  8,  7, 6, 4, 7, 7,  3, 1, 1, 0, 56),
    row("Zaatar",               "herb_spice","levantine",      "herbal",         1, 4, 3, 1, 4, 3, 3,  4, 0, 2,  8,  7, 6, 5, 7, 8,  2, 0, 1, 0, 52),
    row("Labneh",               "cheese",  "levantine",        "bright_acidic",  2, 4, 4, 0, 4, 1, 6,  0, 2, 8,  6,  7, 6, 5, 7, 6,  3, 0, 0, 0, 54),
    row("Sumac",                "herb_spice","levantine",      "bright_acidic",  1, 3, 2, 0, 6, 2, 1,  3, 0, 2,  7,  7, 6, 5, 7, 8,  2, 0, 1, 0, 42),
    row("Kalamata Olive",       "produce", "greek",            "savory",         0, 8, 5, 0, 4, 3, 5,  2, 0, 4, 10,  5, 4, 1, 5, 25, 2, 1, 1, 0, 14),
    row("Smoked Salmon",        "seafood", "nordic",           "umami_bomb",     1, 7, 7, 0, 2, 1, 6,  0, 0, 6,  5,  7, 5, 1, 7, 20, 4, 0, 0, 1, 28),
    row("Shrimp",               "seafood", "italian_american", "savory",         2, 4, 6, 0, 2, 1, 3,  3, 0, 6,  9,  6, 5, 1, 6, 20, 3, 0, 0, 0, 22),
    row("Clam",                 "seafood", "new_haven",        "umami_bomb",     1, 6, 8, 0, 3, 1, 3,  2, 0, 6,  9,  5, 4, 1, 5, 40, 3, 0, 0, 0, 24),
    row("BBQ Chicken",          "meat",    "american",         "sweet",          5, 5, 5, 2, 4, 1, 4,  2, 0, 6,  9,  6, 5, 1, 6, 20, 2, 0, 0, 0, 26),
    row("Meatball",             "meat",    "italian_american", "savory",         2, 5, 6, 1, 2, 1, 6,  3, 0, 6,  9,  5, 4, 1, 5, 40, 2, 0, 0, 0, 18),
    row("Sunny-Side Egg",       "protein", "french_italian",   "umami_bomb",     1, 3, 5, 0, 1, 1, 6,  0, 0, 8,  6,  8, 7, 4, 8, 10, 2, 0, 0, 1, 40),
    row("Blue Cheese Crumble",  "cheese",  "european",         "pungent",        1, 8, 6, 0, 3, 4, 7,  2, 6, 4,  8,  6, 5, 1, 7, 20, 3, 0, 0, 1, 22),
    row("Fried Shallot",        "herb_spice","southeast_asian","savory",         3, 4, 5, 0, 1, 1, 5,  8, 0, 1,  9,  7, 6, 2, 7, 3,  2, 0, 1, 0, 50),
    row("Sesame Seeds",         "herb_spice","asian",          "savory",         2, 2, 3, 0, 1, 2, 3,  5, 0, 1,  9,  5, 4, 2, 5, 5,  1, 0, 1, 0, 32),
    row("Everything Bagel Mix", "herb_spice","american",       "savory",         1, 6, 4, 0, 1, 1, 2,  5, 0, 1,  9,  8, 7, 4, 8, 3,  2, 0, 1, 0, 58),
    row("Vindaloo Pork",        "meat",    "indian",           "spicy",          3, 5, 6, 8, 5, 1, 6,  2, 0, 7,  9,  8, 7, 5, 8, 5,  3, 0, 0, 1, 70),
]


# ---------------------------------------------------------------------------
# TEST: emerging + speculative/"undiscovered" candidates (trend_score blank)
# ---------------------------------------------------------------------------

TEST = [
    row("Ube Cream",                "sauce",   "filipino",       "sweet",          6, 1, 2, 0, 2, 2, 5,  0, 0, 7,  7, 10, 10, 1,10, 2, 4, 0, 1, 1, ""),
    row("Miso Butter",              "sauce",   "japanese",       "umami_bomb",     3, 6, 9, 0, 2, 1, 8,  0, 2, 7,  8,  9,  9, 3, 9, 4, 3, 0, 0, 1, ""),
    row("Dashi-Glazed Shiitake",    "produce", "japanese",       "umami_bomb",     3, 5, 9, 0, 2, 1, 3,  2, 0, 6,  8,  8,  8, 4, 8, 3, 4, 0, 1, 1, ""),
    row("Gochujang Honey Glaze",    "sauce",   "korean_fusion",  "sweet",          6, 5, 7, 5, 3, 1, 3,  0, 0, 8,  8,  9,  9, 6, 9, 2, 3, 1, 0, 0, ""),
    row("Black Garlic Confit",      "produce", "asian",          "umami_bomb",     4, 3, 8, 0, 2, 2, 3,  1, 0, 6,  8,  8,  8, 5, 8, 3, 5, 1, 1, 1, ""),
    row("Smoked Trout Roe",         "seafood", "nordic",         "umami_bomb",     1, 8, 8, 0, 3, 1, 6,  3, 0, 7,  6,  7,  7, 5, 9, 4, 5, 0, 0, 1, ""),
    row("Fig + Gorgonzola Combo",   "combo",   "italian",        "sweet",          7, 6, 6, 0, 3, 3, 6,  0, 6, 8,  7,  9,  8, 2, 9, 8, 4, 1, 0, 1, ""),
    row("Jackfruit Carnitas",       "meat",    "modern_plant",   "sweet",          4, 4, 4, 2, 3, 1, 2,  2, 0, 7,  7,  7,  7, 5, 7, 4, 3, 0, 1, 0, ""),
    row("Matcha Cream (dessert)",   "sauce",   "japanese",       "herbal",         4, 1, 2, 0, 1, 4, 5,  0, 0, 7,  6,  8,  8, 5, 9, 3, 4, 0, 1, 1, ""),
    row("Labneh + Cucumber + Mint", "combo",   "levantine",      "bright_acidic",  2, 4, 3, 0, 4, 2, 5,  3, 0, 8,  6,  7,  7, 5, 8, 5, 3, 0, 0, 0, ""),
    row("Harissa Lamb",             "meat",    "moroccan",       "spicy",          2, 5, 7, 7, 3, 1, 7,  2, 0, 5,  9,  8,  8, 6, 8, 5, 4, 0, 0, 1, ""),
    row("Sumac + Pomegranate",      "combo",   "levantine",      "bright_acidic",  5, 3, 3, 0, 6, 2, 1,  4, 0, 7,  7,  8,  7, 5, 8, 4, 4, 0, 1, 0, ""),
    row("Furikake + Nori",          "herb_spice","japanese",     "umami_bomb",     1, 6, 8, 0, 2, 2, 2,  5, 0, 1,  8,  7,  7, 5, 8, 3, 3, 0, 1, 0, ""),
    row("Bonito Flakes",            "seafood", "japanese",       "umami_bomb",     1, 6, 9, 0, 2, 1, 2,  4, 0, 1,  3,  8,  8, 7, 8, 5, 4, 0, 0, 1, ""),
    row("Yuzu Kosho",               "sauce",   "japanese",       "bright_acidic",  1, 5, 4, 5, 7, 1, 1,  0, 0, 7,  7,  8,  8, 7, 8, 4, 4, 1, 0, 1, ""),
    row("Tamarind Glaze",           "sauce",   "indian",         "sweet",          6, 2, 3, 2, 7, 2, 1,  0, 0, 7,  7,  7,  7, 5, 7, 3, 2, 1, 1, 0, ""),
    row("Curry Leaf Tempering",     "herb_spice","indian",       "herbal",         1, 2, 3, 1, 2, 3, 2,  3, 0, 2,  6,  7,  7, 6, 7, 3, 4, 0, 1, 0, ""),
    row("Sichuan Peppercorn",       "herb_spice","chinese",      "spicy",          1, 2, 3, 6, 3, 3, 1,  4, 0, 1,  5,  8,  7, 7, 7, 3, 5, 0, 1, 0, ""),
    row("Aji Amarillo Paste",       "sauce",   "peruvian",       "spicy",          3, 4, 5, 6, 4, 1, 2,  0, 0, 7,  8,  7,  7, 7, 7, 3, 5, 1, 0, 0, ""),
    row("Huacatay (Black Mint)",    "herb_spice","peruvian",     "herbal",         1, 1, 3, 1, 3, 4, 1,  2, 0, 6,  5,  6,  6, 7, 6, 2, 5, 0, 1, 1, ""),
    row("Berbere Spice",            "herb_spice","ethiopian",    "spicy",          1, 3, 4, 6, 2, 3, 2,  4, 0, 2,  7,  7,  6, 7, 7, 3, 4, 0, 1, 0, ""),
    row("Injera Crumbs",            "grain",   "ethiopian",      "funky_fermented",1, 3, 3, 0, 5, 2, 1,  6, 0, 2,  7,  4,  4, 7, 5, 2, 4, 1, 1, 0, ""),
    row("Mole Negro Drizzle",       "sauce",   "mexican",        "umami_bomb",     3, 3, 7, 3, 3, 4, 4,  0, 0, 7,  7,  8,  7, 5, 8, 4, 4, 0, 0, 1, ""),
    row("Chaat Masala",             "herb_spice","indian",       "bright_acidic",  2, 5, 3, 3, 5, 2, 1,  4, 0, 2,  7,  7,  7, 5, 7, 3, 2, 0, 1, 0, ""),
    row("Cilantro Chutney",         "sauce",   "indian",         "herbal",         2, 3, 3, 3, 5, 3, 2,  0, 0, 8,  6,  7,  7, 4, 7, 3, 3, 1, 1, 0, ""),
    row("Paneer Makhani",           "cheese",  "indian",         "umami_bomb",     4, 5, 7, 3, 3, 1, 7,  1, 0, 8,  8,  9,  9, 6, 9, 5, 3, 0, 0, 1, ""),
    row("Keema (Spiced Ground Lamb)","meat",   "indian",         "spicy",          2, 5, 7, 6, 3, 1, 6,  3, 0, 6,  9,  7,  8, 6, 8, 4, 3, 0, 0, 0, ""),
    row("Methi (Fenugreek) Leaf",   "herb_spice","indian",       "herbal",         1, 1, 3, 0, 1, 5, 1,  3, 0, 5,  5,  6,  6, 7, 6, 3, 5, 0, 1, 0, ""),
    row("Amchur (Dried Mango)",     "herb_spice","indian",       "bright_acidic",  3, 2, 2, 0, 7, 2, 1,  3, 0, 2,  6,  6,  5, 6, 6, 3, 4, 0, 1, 0, ""),
    row("Tamago (Sweet Omelet)",    "protein", "japanese",       "sweet",          6, 3, 5, 0, 1, 0, 5,  1, 0, 7,  5,  7,  7, 6, 8, 3, 3, 0, 0, 1, ""),
    row("Crispy Rice Paper",        "grain",   "vietnamese",     "savory",         1, 2, 2, 0, 1, 1, 1,  9, 0, 1,  6,  6,  6, 6, 7, 2, 3, 0, 1, 0, ""),
    row("Pandan Cream",             "sauce",   "southeast_asian","herbal",         5, 1, 2, 0, 1, 2, 5,  0, 0, 7,  6,  7,  7, 6, 8, 2, 5, 0, 1, 1, ""),
    row("Green Papaya Relish",      "produce", "thai",           "bright_acidic",  3, 3, 3, 4, 6, 2, 1,  5, 0, 7,  5,  7,  7, 6, 8, 2, 4, 0, 1, 0, ""),
    row("Chicken Adobo",            "meat",    "filipino",       "funky_fermented",2, 6, 7, 2, 5, 1, 5,  2, 0, 7,  9,  7,  7, 5, 7, 3, 4, 1, 0, 0, ""),
    row("Longganisa",               "meat",    "filipino",       "sweet",          5, 5, 6, 2, 2, 1, 7,  3, 0, 4,  9,  7,  7, 6, 7, 3, 5, 0, 0, 0, ""),
    row("Bottarga (Cured Roe)",     "seafood", "sardinian",      "umami_bomb",     1, 9, 9, 0, 3, 2, 5,  4, 0, 2,  8,  8,  7, 5, 9, 6, 5, 0, 0, 1, ""),
    row("Confit Egg Yolk",          "protein", "french",         "umami_bomb",     1, 3, 7, 0, 1, 0, 8,  0, 0, 9,  4,  8,  8, 5, 9, 3, 4, 0, 0, 1, ""),
    row("Koji-Aged Bacon",          "meat",    "japanese_american","umami_bomb",   2, 7, 9, 1, 2, 1, 8,  6, 0, 3, 10,  9,  9, 7, 9, 2, 5, 1, 0, 1, ""),
    row("Black Lime (Loomi)",       "herb_spice","persian",      "bright_acidic",  1, 3, 2, 0, 7, 5, 1,  3, 0, 1,  5,  7,  6, 7, 7, 2, 5, 0, 1, 1, ""),
    row("Ajvar",                    "sauce",   "balkan",         "sweet",          4, 3, 5, 3, 3, 2, 3,  0, 0, 7,  7,  6,  6, 6, 7, 3, 4, 0, 1, 0, ""),
    row("Urfa Biber",               "herb_spice","turkish",      "smoky",          2, 3, 4, 5, 2, 4, 3,  3, 0, 2,  8,  7,  7, 7, 7, 3, 4, 0, 1, 0, ""),
    row("Maras Pepper",             "herb_spice","turkish",      "spicy",          2, 3, 3, 5, 2, 2, 2,  3, 0, 2,  8,  7,  6, 6, 7, 3, 3, 0, 1, 0, ""),
    row("Chili Crisp + Honey",      "sauce",   "fusion",         "sweet",          7, 5, 7, 6, 3, 1, 5,  5, 0, 6,  9,  9,  9, 5, 9, 2, 3, 0, 0, 0, ""),
    row("Boquerones (White Anch.)", "seafood", "spanish",        "bright_acidic",  1, 7, 6, 0, 6, 1, 5,  1, 0, 7,  6,  7,  6, 4, 8, 6, 4, 1, 0, 1, ""),
    row("Manchurian Sauce",         "sauce",   "indo_chinese",   "umami_bomb",     4, 5, 7, 5, 3, 1, 3,  0, 0, 7,  8,  7,  7, 6, 7, 3, 4, 0, 0, 0, ""),
    row("Nopales (Cactus)",         "produce", "mexican",        "bright_acidic",  1, 2, 3, 0, 4, 2, 1,  3, 0, 7,  7,  6,  6, 6, 6, 3, 5, 0, 1, 0, ""),
    row("Chimichurri",              "sauce",   "argentinian",    "herbal",         1, 3, 3, 2, 5, 3, 5,  0, 0, 8,  6,  7,  7, 3, 7, 2, 3, 0, 1, 0, ""),
    row("Pickled Wakame",           "produce", "japanese",       "funky_fermented",2, 6, 6, 0, 5, 2, 1,  2, 0, 8,  6,  7,  7, 6, 8, 3, 5, 1, 1, 0, ""),
    row("Ssamjang",                 "sauce",   "korean",         "umami_bomb",     3, 6, 7, 4, 3, 2, 3,  0, 0, 7,  7,  8,  8, 6, 8, 3, 4, 1, 0, 0, ""),
    row("Perilla Leaf",             "herb_spice","korean",       "herbal",         1, 1, 2, 1, 2, 3, 1,  2, 0, 7,  5,  6,  6, 6, 6, 2, 5, 0, 1, 0, ""),
    row("Lemongrass-Ginger Relish", "herb_spice","southeast_asian","bright_acidic",2, 2, 3, 2, 5, 2, 2,  3, 0, 6,  6,  7,  7, 6, 7, 3, 5, 0, 1, 0, ""),
    row("Ikura (Salmon Roe)",       "seafood", "japanese",       "umami_bomb",     1, 6, 7, 0, 3, 1, 5,  3, 0, 9,  4,  7,  7, 5, 9, 4, 5, 0, 0, 1, ""),
    row("Truffle Pecorino",         "cheese",  "italian",        "umami_bomb",     1, 8, 9, 0, 2, 1, 7,  3, 3, 3, 10,  8,  8, 4, 9, 6, 5, 0, 0, 1, ""),
    row("Stracciatella",            "cheese",  "italian",        "savory",         2, 4, 6, 0, 1, 0, 8,  0, 6, 9,  5,  8,  7, 1, 9, 6, 4, 0, 0, 1, ""),
    row("Brown Butter + Sage",      "sauce",   "italian",        "umami_bomb",     2, 3, 6, 0, 1, 2, 9,  0, 0, 6,  7,  8,  7, 3, 8, 4, 3, 0, 0, 1, ""),
    row("Candied Kumquat",          "fruit",   "asian",          "sweet",          9, 1, 2, 0, 5, 2, 1,  3, 0, 5,  7,  8,  8, 5, 9, 2, 4, 1, 1, 1, ""),
]


FLAVOR_FAMILIES = [
    "savory", "umami_bomb", "bright_acidic", "spicy", "sweet",
    "herbal", "smoky", "pungent", "funky_fermented",
]
ONE_HOT_COLS = [f"is_family_{f}" for f in FLAVOR_FAMILIES]


def expand_one_hot(rows: list[list]) -> tuple[list[str], list[list]]:
    fam_idx = COLS.index("flavor_family")
    trend_idx = COLS.index("trend_score")
    new_cols = COLS[:trend_idx] + ONE_HOT_COLS + [COLS[trend_idx]]
    new_rows = []
    for r in rows:
        fam = r[fam_idx]
        one_hot = [1 if f == fam else 0 for f in FLAVOR_FAMILIES]
        new_rows.append(r[:trend_idx] + one_hot + [r[trend_idx]])
    return new_cols, new_rows


def write_csv(path: Path, rows: list[list]) -> None:
    # XGBoost ranking requires rows within the same group (category) to be
    # contiguous. Sort by category, then by name for determinism.
    cat_idx = COLS.index("category")
    name_idx = COLS.index("name")
    rows_sorted = sorted(rows, key=lambda r: (r[cat_idx], r[name_idx]))
    header, rows_expanded = expand_one_hot(rows_sorted)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows_expanded)


if __name__ == "__main__":
    here = Path(__file__).parent
    write_csv(here / "data" / "train.csv", TRAIN)
    write_csv(here / "data" / "test.csv", TEST)
    print(f"Wrote {len(TRAIN)} training rows, {len(TEST)} test rows.")
