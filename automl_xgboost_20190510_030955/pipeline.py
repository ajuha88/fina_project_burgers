import pandas as pd
import numpy as np
import json
import os
import csv
import sys
import warnings
from datetime import datetime
from math import floor
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb


def build_encoders(df):
    """Builds encoders for fields to be used when
    processing data for the model.

    All encoder specifications are stored in locally
    in /encoders as .json files.

    # Arguments
        df: A pandas DataFrame containing the data.
    """

    # Unnamed: 0
    unnamed_0_enc = df['Unnamed: 0']
    unnamed_0_bins = unnamed_0_enc.quantile(np.linspace(0, 1, 4+1))

    with open(os.path.join('encoders', 'unnamed_0_bins.json'),
              'w', encoding='utf8') as outfile:
        json.dump(unnamed_0_bins.tolist(), outfile, ensure_ascii=False)

    # American Restaurant
    american_restaurant_tf = df['American Restaurant'].values
    american_restaurant_encoder = LabelBinarizer()
    american_restaurant_encoder.fit(american_restaurant_tf)

    with open(os.path.join('encoders', 'american_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(american_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Asian Restaurant
    asian_restaurant_tf = df['Asian Restaurant'].values
    asian_restaurant_encoder = LabelBinarizer()
    asian_restaurant_encoder.fit(asian_restaurant_tf)

    with open(os.path.join('encoders', 'asian_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(asian_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Australian Restaurant
    australian_restaurant_tf = df['Australian Restaurant'].values
    australian_restaurant_encoder = LabelBinarizer()
    australian_restaurant_encoder.fit(australian_restaurant_tf)

    with open(os.path.join('encoders', 'australian_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(australian_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # BBQ Joint
    bbq_joint_tf = df['BBQ Joint'].values
    bbq_joint_encoder = LabelBinarizer()
    bbq_joint_encoder.fit(bbq_joint_tf)

    with open(os.path.join('encoders', 'bbq_joint_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bbq_joint_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Bakery
    bakery_tf = df['Bakery'].values
    bakery_encoder = LabelBinarizer()
    bakery_encoder.fit(bakery_tf)

    with open(os.path.join('encoders', 'bakery_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bakery_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Bar
    bar_tf = df['Bar'].values
    bar_encoder = LabelBinarizer()
    bar_encoder.fit(bar_tf)

    with open(os.path.join('encoders', 'bar_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bar_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Beer Garden
    beer_garden_tf = df['Beer Garden'].values
    beer_garden_encoder = LabelBinarizer()
    beer_garden_encoder.fit(beer_garden_tf)

    with open(os.path.join('encoders', 'beer_garden_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(beer_garden_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Breakfast Spot
    breakfast_spot_tf = df['Breakfast Spot'].values
    breakfast_spot_encoder = LabelBinarizer()
    breakfast_spot_encoder.fit(breakfast_spot_tf)

    with open(os.path.join('encoders', 'breakfast_spot_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(breakfast_spot_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Brewery
    brewery_tf = df['Brewery'].values
    brewery_encoder = LabelBinarizer()
    brewery_encoder.fit(brewery_tf)

    with open(os.path.join('encoders', 'brewery_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(brewery_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Burger Joint
    burger_joint_tf = df['Burger Joint'].values
    burger_joint_encoder = LabelBinarizer()
    burger_joint_encoder.fit(burger_joint_tf)

    with open(os.path.join('encoders', 'burger_joint_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(burger_joint_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Café
    café_tf = df['Café'].values
    café_encoder = LabelBinarizer()
    café_encoder.fit(café_tf)

    with open(os.path.join('encoders', 'café_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(café_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Cocktail Bar
    cocktail_bar_tf = df['Cocktail Bar'].values
    cocktail_bar_encoder = LabelBinarizer()
    cocktail_bar_encoder.fit(cocktail_bar_tf)

    with open(os.path.join('encoders', 'cocktail_bar_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(cocktail_bar_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Coffee Shop
    coffee_shop_tf = df['Coffee Shop'].values
    coffee_shop_encoder = LabelBinarizer()
    coffee_shop_encoder.fit(coffee_shop_tf)

    with open(os.path.join('encoders', 'coffee_shop_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(coffee_shop_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Deli / Bodega
    deli_bodega_tf = df['Deli / Bodega'].values
    deli_bodega_encoder = LabelBinarizer()
    deli_bodega_encoder.fit(deli_bodega_tf)

    with open(os.path.join('encoders', 'deli_bodega_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(deli_bodega_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Diner
    diner_tf = df['Diner'].values
    diner_encoder = LabelBinarizer()
    diner_encoder.fit(diner_tf)

    with open(os.path.join('encoders', 'diner_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(diner_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Dive Bar
    dive_bar_tf = df['Dive Bar'].values
    dive_bar_encoder = LabelBinarizer()
    dive_bar_encoder.fit(dive_bar_tf)

    with open(os.path.join('encoders', 'dive_bar_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(dive_bar_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Eastern European Restaurant
    eastern_european_restaurant_tf = df['Eastern European Restaurant'].values
    eastern_european_restaurant_encoder = LabelBinarizer()
    eastern_european_restaurant_encoder.fit(eastern_european_restaurant_tf)

    with open(os.path.join('encoders', 'eastern_european_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(eastern_european_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Fast Food Restaurant
    fast_food_restaurant_tf = df['Fast Food Restaurant'].values
    fast_food_restaurant_encoder = LabelBinarizer()
    fast_food_restaurant_encoder.fit(fast_food_restaurant_tf)

    with open(os.path.join('encoders', 'fast_food_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(fast_food_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Food & Drink Shop
    food_drink_shop_tf = df['Food & Drink Shop'].values
    food_drink_shop_encoder = LabelBinarizer()
    food_drink_shop_encoder.fit(food_drink_shop_tf)

    with open(os.path.join('encoders', 'food_drink_shop_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(food_drink_shop_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Food Truck
    food_truck_tf = df['Food Truck'].values
    food_truck_encoder = LabelBinarizer()
    food_truck_encoder.fit(food_truck_tf)

    with open(os.path.join('encoders', 'food_truck_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(food_truck_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # French Restaurant
    french_restaurant_tf = df['French Restaurant'].values
    french_restaurant_encoder = LabelBinarizer()
    french_restaurant_encoder.fit(french_restaurant_tf)

    with open(os.path.join('encoders', 'french_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(french_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Fried Chicken Joint
    fried_chicken_joint_tf = df['Fried Chicken Joint'].values
    fried_chicken_joint_encoder = LabelBinarizer()
    fried_chicken_joint_encoder.fit(fried_chicken_joint_tf)

    with open(os.path.join('encoders', 'fried_chicken_joint_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(fried_chicken_joint_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Gastropub
    gastropub_tf = df['Gastropub'].values
    gastropub_encoder = LabelBinarizer()
    gastropub_encoder.fit(gastropub_tf)

    with open(os.path.join('encoders', 'gastropub_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(gastropub_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Greek Restaurant
    greek_restaurant_tf = df['Greek Restaurant'].values
    greek_restaurant_encoder = LabelBinarizer()
    greek_restaurant_encoder.fit(greek_restaurant_tf)

    with open(os.path.join('encoders', 'greek_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(greek_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Hookah Bar
    hookah_bar_tf = df['Hookah Bar'].values
    hookah_bar_encoder = LabelBinarizer()
    hookah_bar_encoder.fit(hookah_bar_tf)

    with open(os.path.join('encoders', 'hookah_bar_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(hookah_bar_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Hot Dog Joint
    hot_dog_joint_tf = df['Hot Dog Joint'].values
    hot_dog_joint_encoder = LabelBinarizer()
    hot_dog_joint_encoder.fit(hot_dog_joint_tf)

    with open(os.path.join('encoders', 'hot_dog_joint_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(hot_dog_joint_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Hotel Bar
    hotel_bar_tf = df['Hotel Bar'].values
    hotel_bar_encoder = LabelBinarizer()
    hotel_bar_encoder.fit(hotel_bar_tf)

    with open(os.path.join('encoders', 'hotel_bar_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(hotel_bar_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Irish Pub
    irish_pub_tf = df['Irish Pub'].values
    irish_pub_encoder = LabelBinarizer()
    irish_pub_encoder.fit(irish_pub_tf)

    with open(os.path.join('encoders', 'irish_pub_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(irish_pub_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Italian Restaurant
    italian_restaurant_tf = df['Italian Restaurant'].values
    italian_restaurant_encoder = LabelBinarizer()
    italian_restaurant_encoder.fit(italian_restaurant_tf)

    with open(os.path.join('encoders', 'italian_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(italian_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Juice Bar
    juice_bar_tf = df['Juice Bar'].values
    juice_bar_encoder = LabelBinarizer()
    juice_bar_encoder.fit(juice_bar_tf)

    with open(os.path.join('encoders', 'juice_bar_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(juice_bar_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Lounge
    lounge_tf = df['Lounge'].values
    lounge_encoder = LabelBinarizer()
    lounge_encoder.fit(lounge_tf)

    with open(os.path.join('encoders', 'lounge_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(lounge_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Mediterranean Restaurant
    mediterranean_restaurant_tf = df['Mediterranean Restaurant'].values
    mediterranean_restaurant_encoder = LabelBinarizer()
    mediterranean_restaurant_encoder.fit(mediterranean_restaurant_tf)

    with open(os.path.join('encoders', 'mediterranean_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(mediterranean_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # New American Restaurant
    new_american_restaurant_tf = df['New American Restaurant'].values
    new_american_restaurant_encoder = LabelBinarizer()
    new_american_restaurant_encoder.fit(new_american_restaurant_tf)

    with open(os.path.join('encoders', 'new_american_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(new_american_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Piano Bar
    piano_bar_tf = df['Piano Bar'].values
    piano_bar_encoder = LabelBinarizer()
    piano_bar_encoder.fit(piano_bar_tf)

    with open(os.path.join('encoders', 'piano_bar_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(piano_bar_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Pizza Place
    pizza_place_tf = df['Pizza Place'].values
    pizza_place_encoder = LabelBinarizer()
    pizza_place_encoder.fit(pizza_place_tf)

    with open(os.path.join('encoders', 'pizza_place_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(pizza_place_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Pub
    pub_tf = df['Pub'].values
    pub_encoder = LabelBinarizer()
    pub_encoder.fit(pub_tf)

    with open(os.path.join('encoders', 'pub_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(pub_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Restaurant
    restaurant_tf = df['Restaurant'].values
    restaurant_encoder = LabelBinarizer()
    restaurant_encoder.fit(restaurant_tf)

    with open(os.path.join('encoders', 'restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Rock Club
    rock_club_tf = df['Rock Club'].values
    rock_club_encoder = LabelBinarizer()
    rock_club_encoder.fit(rock_club_tf)

    with open(os.path.join('encoders', 'rock_club_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(rock_club_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Sandwich Place
    sandwich_place_tf = df['Sandwich Place'].values
    sandwich_place_encoder = LabelBinarizer()
    sandwich_place_encoder.fit(sandwich_place_tf)

    with open(os.path.join('encoders', 'sandwich_place_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(sandwich_place_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Seafood Restaurant
    seafood_restaurant_tf = df['Seafood Restaurant'].values
    seafood_restaurant_encoder = LabelBinarizer()
    seafood_restaurant_encoder.fit(seafood_restaurant_tf)

    with open(os.path.join('encoders', 'seafood_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(seafood_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Southern / Soul Food Restaurant
    southern_soul_food_restaurant_tf = df['Southern / Soul Food Restaurant'].values
    southern_soul_food_restaurant_encoder = LabelBinarizer()
    southern_soul_food_restaurant_encoder.fit(southern_soul_food_restaurant_tf)

    with open(os.path.join('encoders', 'southern_soul_food_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(southern_soul_food_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Sports Bar
    sports_bar_tf = df['Sports Bar'].values
    sports_bar_encoder = LabelBinarizer()
    sports_bar_encoder.fit(sports_bar_tf)

    with open(os.path.join('encoders', 'sports_bar_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(sports_bar_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Steakhouse
    steakhouse_tf = df['Steakhouse'].values
    steakhouse_encoder = LabelBinarizer()
    steakhouse_encoder.fit(steakhouse_tf)

    with open(os.path.join('encoders', 'steakhouse_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(steakhouse_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Theme Restaurant
    theme_restaurant_tf = df['Theme Restaurant'].values
    theme_restaurant_encoder = LabelBinarizer()
    theme_restaurant_encoder.fit(theme_restaurant_tf)

    with open(os.path.join('encoders', 'theme_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(theme_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Vegetarian / Vegan Restaurant
    vegetarian_vegan_restaurant_tf = df['Vegetarian / Vegan Restaurant'].values
    vegetarian_vegan_restaurant_encoder = LabelBinarizer()
    vegetarian_vegan_restaurant_encoder.fit(vegetarian_vegan_restaurant_tf)

    with open(os.path.join('encoders', 'vegetarian_vegan_restaurant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(vegetarian_vegan_restaurant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Whisky Bar
    whisky_bar_tf = df['Whisky Bar'].values
    whisky_bar_encoder = LabelBinarizer()
    whisky_bar_encoder.fit(whisky_bar_tf)

    with open(os.path.join('encoders', 'whisky_bar_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(whisky_bar_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Wings Joint
    wings_joint_tf = df['Wings Joint'].values
    wings_joint_encoder = LabelBinarizer()
    wings_joint_encoder.fit(wings_joint_tf)

    with open(os.path.join('encoders', 'wings_joint_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(wings_joint_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Bath Beach
    bath_beach_tf = df['Bath Beach'].values
    bath_beach_encoder = LabelBinarizer()
    bath_beach_encoder.fit(bath_beach_tf)

    with open(os.path.join('encoders', 'bath_beach_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bath_beach_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Bay Ridge
    bay_ridge_tf = df['Bay Ridge'].values
    bay_ridge_encoder = LabelBinarizer()
    bay_ridge_encoder.fit(bay_ridge_tf)

    with open(os.path.join('encoders', 'bay_ridge_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bay_ridge_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Baychester
    baychester_tf = df['Baychester'].values
    baychester_encoder = LabelBinarizer()
    baychester_encoder.fit(baychester_tf)

    with open(os.path.join('encoders', 'baychester_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(baychester_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Bedford Park
    bedford_park_tf = df['Bedford Park'].values
    bedford_park_encoder = LabelBinarizer()
    bedford_park_encoder.fit(bedford_park_tf)

    with open(os.path.join('encoders', 'bedford_park_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bedford_park_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Bedford Stuyvesant
    bedford_stuyvesant_tf = df['Bedford Stuyvesant'].values
    bedford_stuyvesant_encoder = LabelBinarizer()
    bedford_stuyvesant_encoder.fit(bedford_stuyvesant_tf)

    with open(os.path.join('encoders', 'bedford_stuyvesant_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bedford_stuyvesant_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Belmont
    belmont_tf = df['Belmont'].values
    belmont_encoder = LabelBinarizer()
    belmont_encoder.fit(belmont_tf)

    with open(os.path.join('encoders', 'belmont_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(belmont_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Boerum Hill
    boerum_hill_tf = df['Boerum Hill'].values
    boerum_hill_encoder = LabelBinarizer()
    boerum_hill_encoder.fit(boerum_hill_tf)

    with open(os.path.join('encoders', 'boerum_hill_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(boerum_hill_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Brighton Beach
    brighton_beach_tf = df['Brighton Beach'].values
    brighton_beach_encoder = LabelBinarizer()
    brighton_beach_encoder.fit(brighton_beach_tf)

    with open(os.path.join('encoders', 'brighton_beach_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(brighton_beach_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Brooklyn Heights
    brooklyn_heights_tf = df['Brooklyn Heights'].values
    brooklyn_heights_encoder = LabelBinarizer()
    brooklyn_heights_encoder.fit(brooklyn_heights_tf)

    with open(os.path.join('encoders', 'brooklyn_heights_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(brooklyn_heights_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Bushwick
    bushwick_tf = df['Bushwick'].values
    bushwick_encoder = LabelBinarizer()
    bushwick_encoder.fit(bushwick_tf)

    with open(os.path.join('encoders', 'bushwick_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(bushwick_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Carroll Gardens
    carroll_gardens_tf = df['Carroll Gardens'].values
    carroll_gardens_encoder = LabelBinarizer()
    carroll_gardens_encoder.fit(carroll_gardens_tf)

    with open(os.path.join('encoders', 'carroll_gardens_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(carroll_gardens_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Central Harlem
    central_harlem_tf = df['Central Harlem'].values
    central_harlem_encoder = LabelBinarizer()
    central_harlem_encoder.fit(central_harlem_tf)

    with open(os.path.join('encoders', 'central_harlem_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(central_harlem_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Chinatown
    chinatown_tf = df['Chinatown'].values
    chinatown_encoder = LabelBinarizer()
    chinatown_encoder.fit(chinatown_tf)

    with open(os.path.join('encoders', 'chinatown_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(chinatown_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # City Island
    city_island_tf = df['City Island'].values
    city_island_encoder = LabelBinarizer()
    city_island_encoder.fit(city_island_tf)

    with open(os.path.join('encoders', 'city_island_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(city_island_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # City Line
    city_line_tf = df['City Line'].values
    city_line_encoder = LabelBinarizer()
    city_line_encoder.fit(city_line_tf)

    with open(os.path.join('encoders', 'city_line_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(city_line_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Clinton
    clinton_tf = df['Clinton'].values
    clinton_encoder = LabelBinarizer()
    clinton_encoder.fit(clinton_tf)

    with open(os.path.join('encoders', 'clinton_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(clinton_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Clinton Hill
    clinton_hill_tf = df['Clinton Hill'].values
    clinton_hill_encoder = LabelBinarizer()
    clinton_hill_encoder.fit(clinton_hill_tf)

    with open(os.path.join('encoders', 'clinton_hill_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(clinton_hill_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Co-op City
    co_op_city_tf = df['Co-op City'].values
    co_op_city_encoder = LabelBinarizer()
    co_op_city_encoder.fit(co_op_city_tf)

    with open(os.path.join('encoders', 'co_op_city_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(co_op_city_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Cobble Hill
    cobble_hill_tf = df['Cobble Hill'].values
    cobble_hill_encoder = LabelBinarizer()
    cobble_hill_encoder.fit(cobble_hill_tf)

    with open(os.path.join('encoders', 'cobble_hill_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(cobble_hill_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Crown Heights
    crown_heights_tf = df['Crown Heights'].values
    crown_heights_encoder = LabelBinarizer()
    crown_heights_encoder.fit(crown_heights_tf)

    with open(os.path.join('encoders', 'crown_heights_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(crown_heights_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Downtown
    downtown_tf = df['Downtown'].values
    downtown_encoder = LabelBinarizer()
    downtown_encoder.fit(downtown_tf)

    with open(os.path.join('encoders', 'downtown_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(downtown_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Dyker Heights
    dyker_heights_tf = df['Dyker Heights'].values
    dyker_heights_encoder = LabelBinarizer()
    dyker_heights_encoder.fit(dyker_heights_tf)

    with open(os.path.join('encoders', 'dyker_heights_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(dyker_heights_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # East Harlem
    east_harlem_tf = df['East Harlem'].values
    east_harlem_encoder = LabelBinarizer()
    east_harlem_encoder.fit(east_harlem_tf)

    with open(os.path.join('encoders', 'east_harlem_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(east_harlem_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # East Tremont
    east_tremont_tf = df['East Tremont'].values
    east_tremont_encoder = LabelBinarizer()
    east_tremont_encoder.fit(east_tremont_tf)

    with open(os.path.join('encoders', 'east_tremont_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(east_tremont_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # East Williamsburg
    east_williamsburg_tf = df['East Williamsburg'].values
    east_williamsburg_encoder = LabelBinarizer()
    east_williamsburg_encoder.fit(east_williamsburg_tf)

    with open(os.path.join('encoders', 'east_williamsburg_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(east_williamsburg_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Eastchester
    eastchester_tf = df['Eastchester'].values
    eastchester_encoder = LabelBinarizer()
    eastchester_encoder.fit(eastchester_tf)

    with open(os.path.join('encoders', 'eastchester_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(eastchester_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Edgewater Park
    edgewater_park_tf = df['Edgewater Park'].values
    edgewater_park_encoder = LabelBinarizer()
    edgewater_park_encoder.fit(edgewater_park_tf)

    with open(os.path.join('encoders', 'edgewater_park_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(edgewater_park_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Flatlands
    flatlands_tf = df['Flatlands'].values
    flatlands_encoder = LabelBinarizer()
    flatlands_encoder.fit(flatlands_tf)

    with open(os.path.join('encoders', 'flatlands_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(flatlands_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Fordham
    fordham_tf = df['Fordham'].values
    fordham_encoder = LabelBinarizer()
    fordham_encoder.fit(fordham_tf)

    with open(os.path.join('encoders', 'fordham_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(fordham_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Fort Greene
    fort_greene_tf = df['Fort Greene'].values
    fort_greene_encoder = LabelBinarizer()
    fort_greene_encoder.fit(fort_greene_tf)

    with open(os.path.join('encoders', 'fort_greene_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(fort_greene_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Fort Hamilton
    fort_hamilton_tf = df['Fort Hamilton'].values
    fort_hamilton_encoder = LabelBinarizer()
    fort_hamilton_encoder.fit(fort_hamilton_tf)

    with open(os.path.join('encoders', 'fort_hamilton_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(fort_hamilton_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Georgetown
    georgetown_tf = df['Georgetown'].values
    georgetown_encoder = LabelBinarizer()
    georgetown_encoder.fit(georgetown_tf)

    with open(os.path.join('encoders', 'georgetown_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(georgetown_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Gerritsen Beach
    gerritsen_beach_tf = df['Gerritsen Beach'].values
    gerritsen_beach_encoder = LabelBinarizer()
    gerritsen_beach_encoder.fit(gerritsen_beach_tf)

    with open(os.path.join('encoders', 'gerritsen_beach_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(gerritsen_beach_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Gowanus
    gowanus_tf = df['Gowanus'].values
    gowanus_encoder = LabelBinarizer()
    gowanus_encoder.fit(gowanus_tf)

    with open(os.path.join('encoders', 'gowanus_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(gowanus_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Greenpoint
    greenpoint_tf = df['Greenpoint'].values
    greenpoint_encoder = LabelBinarizer()
    greenpoint_encoder.fit(greenpoint_tf)

    with open(os.path.join('encoders', 'greenpoint_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(greenpoint_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Hamilton Heights
    hamilton_heights_tf = df['Hamilton Heights'].values
    hamilton_heights_encoder = LabelBinarizer()
    hamilton_heights_encoder.fit(hamilton_heights_tf)

    with open(os.path.join('encoders', 'hamilton_heights_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(hamilton_heights_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # High  Bridge
    high_bridge_tf = df['High  Bridge'].values
    high_bridge_encoder = LabelBinarizer()
    high_bridge_encoder.fit(high_bridge_tf)

    with open(os.path.join('encoders', 'high_bridge_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(high_bridge_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Inwood
    inwood_tf = df['Inwood'].values
    inwood_encoder = LabelBinarizer()
    inwood_encoder.fit(inwood_tf)

    with open(os.path.join('encoders', 'inwood_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(inwood_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Kensington
    kensington_tf = df['Kensington'].values
    kensington_encoder = LabelBinarizer()
    kensington_encoder.fit(kensington_tf)

    with open(os.path.join('encoders', 'kensington_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(kensington_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Kingsbridge
    kingsbridge_tf = df['Kingsbridge'].values
    kingsbridge_encoder = LabelBinarizer()
    kingsbridge_encoder.fit(kingsbridge_tf)

    with open(os.path.join('encoders', 'kingsbridge_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(kingsbridge_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Lenox Hill
    lenox_hill_tf = df['Lenox Hill'].values
    lenox_hill_encoder = LabelBinarizer()
    lenox_hill_encoder.fit(lenox_hill_tf)

    with open(os.path.join('encoders', 'lenox_hill_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(lenox_hill_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Lincoln Square
    lincoln_square_tf = df['Lincoln Square'].values
    lincoln_square_encoder = LabelBinarizer()
    lincoln_square_encoder.fit(lincoln_square_tf)

    with open(os.path.join('encoders', 'lincoln_square_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(lincoln_square_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Manhattan Beach
    manhattan_beach_tf = df['Manhattan Beach'].values
    manhattan_beach_encoder = LabelBinarizer()
    manhattan_beach_encoder.fit(manhattan_beach_tf)

    with open(os.path.join('encoders', 'manhattan_beach_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(manhattan_beach_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Manhattanville
    manhattanville_tf = df['Manhattanville'].values
    manhattanville_encoder = LabelBinarizer()
    manhattanville_encoder.fit(manhattanville_tf)

    with open(os.path.join('encoders', 'manhattanville_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(manhattanville_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Marble Hill
    marble_hill_tf = df['Marble Hill'].values
    marble_hill_encoder = LabelBinarizer()
    marble_hill_encoder.fit(marble_hill_tf)

    with open(os.path.join('encoders', 'marble_hill_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(marble_hill_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Melrose
    melrose_tf = df['Melrose'].values
    melrose_encoder = LabelBinarizer()
    melrose_encoder.fit(melrose_tf)

    with open(os.path.join('encoders', 'melrose_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(melrose_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Midtown
    midtown_tf = df['Midtown'].values
    midtown_encoder = LabelBinarizer()
    midtown_encoder.fit(midtown_tf)

    with open(os.path.join('encoders', 'midtown_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(midtown_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Morris Park
    morris_park_tf = df['Morris Park'].values
    morris_park_encoder = LabelBinarizer()
    morris_park_encoder.fit(morris_park_tf)

    with open(os.path.join('encoders', 'morris_park_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(morris_park_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Morrisania
    morrisania_tf = df['Morrisania'].values
    morrisania_encoder = LabelBinarizer()
    morrisania_encoder.fit(morrisania_tf)

    with open(os.path.join('encoders', 'morrisania_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(morrisania_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Mott Haven
    mott_haven_tf = df['Mott Haven'].values
    mott_haven_encoder = LabelBinarizer()
    mott_haven_encoder.fit(mott_haven_tf)

    with open(os.path.join('encoders', 'mott_haven_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(mott_haven_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Navy Yard
    navy_yard_tf = df['Navy Yard'].values
    navy_yard_encoder = LabelBinarizer()
    navy_yard_encoder.fit(navy_yard_tf)

    with open(os.path.join('encoders', 'navy_yard_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(navy_yard_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # North Riverdale
    north_riverdale_tf = df['North Riverdale'].values
    north_riverdale_encoder = LabelBinarizer()
    north_riverdale_encoder.fit(north_riverdale_tf)

    with open(os.path.join('encoders', 'north_riverdale_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(north_riverdale_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # North Side
    north_side_tf = df['North Side'].values
    north_side_encoder = LabelBinarizer()
    north_side_encoder.fit(north_side_tf)

    with open(os.path.join('encoders', 'north_side_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(north_side_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Norwood
    norwood_tf = df['Norwood'].values
    norwood_encoder = LabelBinarizer()
    norwood_encoder.fit(norwood_tf)

    with open(os.path.join('encoders', 'norwood_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(norwood_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Park Slope
    park_slope_tf = df['Park Slope'].values
    park_slope_encoder = LabelBinarizer()
    park_slope_encoder.fit(park_slope_tf)

    with open(os.path.join('encoders', 'park_slope_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(park_slope_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Parkchester
    parkchester_tf = df['Parkchester'].values
    parkchester_encoder = LabelBinarizer()
    parkchester_encoder.fit(parkchester_tf)

    with open(os.path.join('encoders', 'parkchester_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(parkchester_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Prospect Heights
    prospect_heights_tf = df['Prospect Heights'].values
    prospect_heights_encoder = LabelBinarizer()
    prospect_heights_encoder.fit(prospect_heights_tf)

    with open(os.path.join('encoders', 'prospect_heights_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(prospect_heights_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Prospect Lefferts Gardens
    prospect_lefferts_gardens_tf = df['Prospect Lefferts Gardens'].values
    prospect_lefferts_gardens_encoder = LabelBinarizer()
    prospect_lefferts_gardens_encoder.fit(prospect_lefferts_gardens_tf)

    with open(os.path.join('encoders', 'prospect_lefferts_gardens_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(prospect_lefferts_gardens_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Prospect Park South
    prospect_park_south_tf = df['Prospect Park South'].values
    prospect_park_south_encoder = LabelBinarizer()
    prospect_park_south_encoder.fit(prospect_park_south_tf)

    with open(os.path.join('encoders', 'prospect_park_south_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(prospect_park_south_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Red Hook
    red_hook_tf = df['Red Hook'].values
    red_hook_encoder = LabelBinarizer()
    red_hook_encoder.fit(red_hook_tf)

    with open(os.path.join('encoders', 'red_hook_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(red_hook_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Schuylerville
    schuylerville_tf = df['Schuylerville'].values
    schuylerville_encoder = LabelBinarizer()
    schuylerville_encoder.fit(schuylerville_tf)

    with open(os.path.join('encoders', 'schuylerville_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(schuylerville_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Sheepshead Bay
    sheepshead_bay_tf = df['Sheepshead Bay'].values
    sheepshead_bay_encoder = LabelBinarizer()
    sheepshead_bay_encoder.fit(sheepshead_bay_tf)

    with open(os.path.join('encoders', 'sheepshead_bay_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(sheepshead_bay_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Soundview
    soundview_tf = df['Soundview'].values
    soundview_encoder = LabelBinarizer()
    soundview_encoder.fit(soundview_tf)

    with open(os.path.join('encoders', 'soundview_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(soundview_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # South Side
    south_side_tf = df['South Side'].values
    south_side_encoder = LabelBinarizer()
    south_side_encoder.fit(south_side_tf)

    with open(os.path.join('encoders', 'south_side_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(south_side_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Spring Creek
    spring_creek_tf = df['Spring Creek'].values
    spring_creek_encoder = LabelBinarizer()
    spring_creek_encoder.fit(spring_creek_tf)

    with open(os.path.join('encoders', 'spring_creek_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(spring_creek_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Throgs Neck
    throgs_neck_tf = df['Throgs Neck'].values
    throgs_neck_encoder = LabelBinarizer()
    throgs_neck_encoder.fit(throgs_neck_tf)

    with open(os.path.join('encoders', 'throgs_neck_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(throgs_neck_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Upper East Side
    upper_east_side_tf = df['Upper East Side'].values
    upper_east_side_encoder = LabelBinarizer()
    upper_east_side_encoder.fit(upper_east_side_tf)

    with open(os.path.join('encoders', 'upper_east_side_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(upper_east_side_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Upper West Side
    upper_west_side_tf = df['Upper West Side'].values
    upper_west_side_encoder = LabelBinarizer()
    upper_west_side_encoder.fit(upper_west_side_tf)

    with open(os.path.join('encoders', 'upper_west_side_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(upper_west_side_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Van Nest
    van_nest_tf = df['Van Nest'].values
    van_nest_encoder = LabelBinarizer()
    van_nest_encoder.fit(van_nest_tf)

    with open(os.path.join('encoders', 'van_nest_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(van_nest_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Washington Heights
    washington_heights_tf = df['Washington Heights'].values
    washington_heights_encoder = LabelBinarizer()
    washington_heights_encoder.fit(washington_heights_tf)

    with open(os.path.join('encoders', 'washington_heights_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(washington_heights_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Westchester Square
    westchester_square_tf = df['Westchester Square'].values
    westchester_square_encoder = LabelBinarizer()
    westchester_square_encoder.fit(westchester_square_tf)

    with open(os.path.join('encoders', 'westchester_square_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(westchester_square_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Williamsburg
    williamsburg_tf = df['Williamsburg'].values
    williamsburg_encoder = LabelBinarizer()
    williamsburg_encoder.fit(williamsburg_tf)

    with open(os.path.join('encoders', 'williamsburg_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(williamsburg_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Windsor Terrace
    windsor_terrace_tf = df['Windsor Terrace'].values
    windsor_terrace_encoder = LabelBinarizer()
    windsor_terrace_encoder.fit(windsor_terrace_tf)

    with open(os.path.join('encoders', 'windsor_terrace_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(windsor_terrace_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Woodlawn
    woodlawn_tf = df['Woodlawn'].values
    woodlawn_encoder = LabelBinarizer()
    woodlawn_encoder.fit(woodlawn_tf)

    with open(os.path.join('encoders', 'woodlawn_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(woodlawn_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Yorkville
    yorkville_tf = df['Yorkville'].values
    yorkville_encoder = LabelBinarizer()
    yorkville_encoder.fit(yorkville_tf)

    with open(os.path.join('encoders', 'yorkville_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(yorkville_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Tip Count
    tip_count_enc = df['Tip Count']
    tip_count_bins = tip_count_enc.quantile(np.linspace(0, 1, 4+1))

    with open(os.path.join('encoders', 'tip_count_bins.json'),
              'w', encoding='utf8') as outfile:
        json.dump(tip_count_bins.tolist(), outfile, ensure_ascii=False)

    # Like Count
    like_count_tf = df['Like Count'].values
    like_count_encoder = LabelBinarizer()
    like_count_encoder.fit(like_count_tf)

    with open(os.path.join('encoders', 'like_count_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(like_count_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Target Field: Rating


def load_encoders():
    """Loads the encoders built during `build_encoders`.

    # Returns
        encoders: A dict of encoder objects/specs.
    """

    encoders = {}

    # Unnamed: 0
    unnamed_0_encoder = LabelBinarizer()
    unnamed_0_encoder.classes_ = list(range(4))

    with open(os.path.join('encoders', 'unnamed_0_bins.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        unnamed_0_bins = json.load(infile)
    encoders['unnamed_0_bins'] = unnamed_0_bins
    encoders['unnamed_0_encoder'] = unnamed_0_encoder

    # American Restaurant
    american_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'american_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        american_restaurant_encoder.classes_ = json.load(infile)
    encoders['american_restaurant_encoder'] = american_restaurant_encoder

    # Asian Restaurant
    asian_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'asian_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        asian_restaurant_encoder.classes_ = json.load(infile)
    encoders['asian_restaurant_encoder'] = asian_restaurant_encoder

    # Australian Restaurant
    australian_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'australian_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        australian_restaurant_encoder.classes_ = json.load(infile)
    encoders['australian_restaurant_encoder'] = australian_restaurant_encoder

    # BBQ Joint
    bbq_joint_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bbq_joint_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bbq_joint_encoder.classes_ = json.load(infile)
    encoders['bbq_joint_encoder'] = bbq_joint_encoder

    # Bakery
    bakery_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bakery_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bakery_encoder.classes_ = json.load(infile)
    encoders['bakery_encoder'] = bakery_encoder

    # Bar
    bar_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bar_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bar_encoder.classes_ = json.load(infile)
    encoders['bar_encoder'] = bar_encoder

    # Beer Garden
    beer_garden_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'beer_garden_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        beer_garden_encoder.classes_ = json.load(infile)
    encoders['beer_garden_encoder'] = beer_garden_encoder

    # Breakfast Spot
    breakfast_spot_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'breakfast_spot_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        breakfast_spot_encoder.classes_ = json.load(infile)
    encoders['breakfast_spot_encoder'] = breakfast_spot_encoder

    # Brewery
    brewery_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'brewery_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        brewery_encoder.classes_ = json.load(infile)
    encoders['brewery_encoder'] = brewery_encoder

    # Burger Joint
    burger_joint_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'burger_joint_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        burger_joint_encoder.classes_ = json.load(infile)
    encoders['burger_joint_encoder'] = burger_joint_encoder

    # Café
    café_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'café_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        café_encoder.classes_ = json.load(infile)
    encoders['café_encoder'] = café_encoder

    # Cocktail Bar
    cocktail_bar_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'cocktail_bar_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        cocktail_bar_encoder.classes_ = json.load(infile)
    encoders['cocktail_bar_encoder'] = cocktail_bar_encoder

    # Coffee Shop
    coffee_shop_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'coffee_shop_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        coffee_shop_encoder.classes_ = json.load(infile)
    encoders['coffee_shop_encoder'] = coffee_shop_encoder

    # Deli / Bodega
    deli_bodega_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'deli_bodega_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        deli_bodega_encoder.classes_ = json.load(infile)
    encoders['deli_bodega_encoder'] = deli_bodega_encoder

    # Diner
    diner_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'diner_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        diner_encoder.classes_ = json.load(infile)
    encoders['diner_encoder'] = diner_encoder

    # Dive Bar
    dive_bar_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'dive_bar_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        dive_bar_encoder.classes_ = json.load(infile)
    encoders['dive_bar_encoder'] = dive_bar_encoder

    # Eastern European Restaurant
    eastern_european_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'eastern_european_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        eastern_european_restaurant_encoder.classes_ = json.load(infile)
    encoders['eastern_european_restaurant_encoder'] = eastern_european_restaurant_encoder

    # Fast Food Restaurant
    fast_food_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'fast_food_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        fast_food_restaurant_encoder.classes_ = json.load(infile)
    encoders['fast_food_restaurant_encoder'] = fast_food_restaurant_encoder

    # Food & Drink Shop
    food_drink_shop_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'food_drink_shop_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        food_drink_shop_encoder.classes_ = json.load(infile)
    encoders['food_drink_shop_encoder'] = food_drink_shop_encoder

    # Food Truck
    food_truck_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'food_truck_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        food_truck_encoder.classes_ = json.load(infile)
    encoders['food_truck_encoder'] = food_truck_encoder

    # French Restaurant
    french_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'french_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        french_restaurant_encoder.classes_ = json.load(infile)
    encoders['french_restaurant_encoder'] = french_restaurant_encoder

    # Fried Chicken Joint
    fried_chicken_joint_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'fried_chicken_joint_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        fried_chicken_joint_encoder.classes_ = json.load(infile)
    encoders['fried_chicken_joint_encoder'] = fried_chicken_joint_encoder

    # Gastropub
    gastropub_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'gastropub_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        gastropub_encoder.classes_ = json.load(infile)
    encoders['gastropub_encoder'] = gastropub_encoder

    # Greek Restaurant
    greek_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'greek_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        greek_restaurant_encoder.classes_ = json.load(infile)
    encoders['greek_restaurant_encoder'] = greek_restaurant_encoder

    # Hookah Bar
    hookah_bar_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'hookah_bar_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        hookah_bar_encoder.classes_ = json.load(infile)
    encoders['hookah_bar_encoder'] = hookah_bar_encoder

    # Hot Dog Joint
    hot_dog_joint_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'hot_dog_joint_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        hot_dog_joint_encoder.classes_ = json.load(infile)
    encoders['hot_dog_joint_encoder'] = hot_dog_joint_encoder

    # Hotel Bar
    hotel_bar_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'hotel_bar_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        hotel_bar_encoder.classes_ = json.load(infile)
    encoders['hotel_bar_encoder'] = hotel_bar_encoder

    # Irish Pub
    irish_pub_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'irish_pub_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        irish_pub_encoder.classes_ = json.load(infile)
    encoders['irish_pub_encoder'] = irish_pub_encoder

    # Italian Restaurant
    italian_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'italian_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        italian_restaurant_encoder.classes_ = json.load(infile)
    encoders['italian_restaurant_encoder'] = italian_restaurant_encoder

    # Juice Bar
    juice_bar_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'juice_bar_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        juice_bar_encoder.classes_ = json.load(infile)
    encoders['juice_bar_encoder'] = juice_bar_encoder

    # Lounge
    lounge_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'lounge_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        lounge_encoder.classes_ = json.load(infile)
    encoders['lounge_encoder'] = lounge_encoder

    # Mediterranean Restaurant
    mediterranean_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'mediterranean_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        mediterranean_restaurant_encoder.classes_ = json.load(infile)
    encoders['mediterranean_restaurant_encoder'] = mediterranean_restaurant_encoder

    # New American Restaurant
    new_american_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'new_american_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        new_american_restaurant_encoder.classes_ = json.load(infile)
    encoders['new_american_restaurant_encoder'] = new_american_restaurant_encoder

    # Piano Bar
    piano_bar_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'piano_bar_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        piano_bar_encoder.classes_ = json.load(infile)
    encoders['piano_bar_encoder'] = piano_bar_encoder

    # Pizza Place
    pizza_place_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'pizza_place_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        pizza_place_encoder.classes_ = json.load(infile)
    encoders['pizza_place_encoder'] = pizza_place_encoder

    # Pub
    pub_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'pub_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        pub_encoder.classes_ = json.load(infile)
    encoders['pub_encoder'] = pub_encoder

    # Restaurant
    restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        restaurant_encoder.classes_ = json.load(infile)
    encoders['restaurant_encoder'] = restaurant_encoder

    # Rock Club
    rock_club_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'rock_club_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        rock_club_encoder.classes_ = json.load(infile)
    encoders['rock_club_encoder'] = rock_club_encoder

    # Sandwich Place
    sandwich_place_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'sandwich_place_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        sandwich_place_encoder.classes_ = json.load(infile)
    encoders['sandwich_place_encoder'] = sandwich_place_encoder

    # Seafood Restaurant
    seafood_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'seafood_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        seafood_restaurant_encoder.classes_ = json.load(infile)
    encoders['seafood_restaurant_encoder'] = seafood_restaurant_encoder

    # Southern / Soul Food Restaurant
    southern_soul_food_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'southern_soul_food_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        southern_soul_food_restaurant_encoder.classes_ = json.load(infile)
    encoders['southern_soul_food_restaurant_encoder'] = southern_soul_food_restaurant_encoder

    # Sports Bar
    sports_bar_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'sports_bar_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        sports_bar_encoder.classes_ = json.load(infile)
    encoders['sports_bar_encoder'] = sports_bar_encoder

    # Steakhouse
    steakhouse_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'steakhouse_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        steakhouse_encoder.classes_ = json.load(infile)
    encoders['steakhouse_encoder'] = steakhouse_encoder

    # Theme Restaurant
    theme_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'theme_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        theme_restaurant_encoder.classes_ = json.load(infile)
    encoders['theme_restaurant_encoder'] = theme_restaurant_encoder

    # Vegetarian / Vegan Restaurant
    vegetarian_vegan_restaurant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'vegetarian_vegan_restaurant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        vegetarian_vegan_restaurant_encoder.classes_ = json.load(infile)
    encoders['vegetarian_vegan_restaurant_encoder'] = vegetarian_vegan_restaurant_encoder

    # Whisky Bar
    whisky_bar_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'whisky_bar_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        whisky_bar_encoder.classes_ = json.load(infile)
    encoders['whisky_bar_encoder'] = whisky_bar_encoder

    # Wings Joint
    wings_joint_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'wings_joint_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        wings_joint_encoder.classes_ = json.load(infile)
    encoders['wings_joint_encoder'] = wings_joint_encoder

    # Bath Beach
    bath_beach_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bath_beach_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bath_beach_encoder.classes_ = json.load(infile)
    encoders['bath_beach_encoder'] = bath_beach_encoder

    # Bay Ridge
    bay_ridge_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bay_ridge_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bay_ridge_encoder.classes_ = json.load(infile)
    encoders['bay_ridge_encoder'] = bay_ridge_encoder

    # Baychester
    baychester_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'baychester_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        baychester_encoder.classes_ = json.load(infile)
    encoders['baychester_encoder'] = baychester_encoder

    # Bedford Park
    bedford_park_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bedford_park_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bedford_park_encoder.classes_ = json.load(infile)
    encoders['bedford_park_encoder'] = bedford_park_encoder

    # Bedford Stuyvesant
    bedford_stuyvesant_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bedford_stuyvesant_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bedford_stuyvesant_encoder.classes_ = json.load(infile)
    encoders['bedford_stuyvesant_encoder'] = bedford_stuyvesant_encoder

    # Belmont
    belmont_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'belmont_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        belmont_encoder.classes_ = json.load(infile)
    encoders['belmont_encoder'] = belmont_encoder

    # Boerum Hill
    boerum_hill_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'boerum_hill_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        boerum_hill_encoder.classes_ = json.load(infile)
    encoders['boerum_hill_encoder'] = boerum_hill_encoder

    # Brighton Beach
    brighton_beach_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'brighton_beach_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        brighton_beach_encoder.classes_ = json.load(infile)
    encoders['brighton_beach_encoder'] = brighton_beach_encoder

    # Brooklyn Heights
    brooklyn_heights_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'brooklyn_heights_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        brooklyn_heights_encoder.classes_ = json.load(infile)
    encoders['brooklyn_heights_encoder'] = brooklyn_heights_encoder

    # Bushwick
    bushwick_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'bushwick_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        bushwick_encoder.classes_ = json.load(infile)
    encoders['bushwick_encoder'] = bushwick_encoder

    # Carroll Gardens
    carroll_gardens_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'carroll_gardens_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        carroll_gardens_encoder.classes_ = json.load(infile)
    encoders['carroll_gardens_encoder'] = carroll_gardens_encoder

    # Central Harlem
    central_harlem_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'central_harlem_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        central_harlem_encoder.classes_ = json.load(infile)
    encoders['central_harlem_encoder'] = central_harlem_encoder

    # Chinatown
    chinatown_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'chinatown_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        chinatown_encoder.classes_ = json.load(infile)
    encoders['chinatown_encoder'] = chinatown_encoder

    # City Island
    city_island_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'city_island_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        city_island_encoder.classes_ = json.load(infile)
    encoders['city_island_encoder'] = city_island_encoder

    # City Line
    city_line_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'city_line_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        city_line_encoder.classes_ = json.load(infile)
    encoders['city_line_encoder'] = city_line_encoder

    # Clinton
    clinton_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'clinton_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        clinton_encoder.classes_ = json.load(infile)
    encoders['clinton_encoder'] = clinton_encoder

    # Clinton Hill
    clinton_hill_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'clinton_hill_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        clinton_hill_encoder.classes_ = json.load(infile)
    encoders['clinton_hill_encoder'] = clinton_hill_encoder

    # Co-op City
    co_op_city_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'co_op_city_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        co_op_city_encoder.classes_ = json.load(infile)
    encoders['co_op_city_encoder'] = co_op_city_encoder

    # Cobble Hill
    cobble_hill_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'cobble_hill_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        cobble_hill_encoder.classes_ = json.load(infile)
    encoders['cobble_hill_encoder'] = cobble_hill_encoder

    # Crown Heights
    crown_heights_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'crown_heights_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        crown_heights_encoder.classes_ = json.load(infile)
    encoders['crown_heights_encoder'] = crown_heights_encoder

    # Downtown
    downtown_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'downtown_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        downtown_encoder.classes_ = json.load(infile)
    encoders['downtown_encoder'] = downtown_encoder

    # Dyker Heights
    dyker_heights_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'dyker_heights_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        dyker_heights_encoder.classes_ = json.load(infile)
    encoders['dyker_heights_encoder'] = dyker_heights_encoder

    # East Harlem
    east_harlem_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'east_harlem_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        east_harlem_encoder.classes_ = json.load(infile)
    encoders['east_harlem_encoder'] = east_harlem_encoder

    # East Tremont
    east_tremont_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'east_tremont_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        east_tremont_encoder.classes_ = json.load(infile)
    encoders['east_tremont_encoder'] = east_tremont_encoder

    # East Williamsburg
    east_williamsburg_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'east_williamsburg_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        east_williamsburg_encoder.classes_ = json.load(infile)
    encoders['east_williamsburg_encoder'] = east_williamsburg_encoder

    # Eastchester
    eastchester_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'eastchester_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        eastchester_encoder.classes_ = json.load(infile)
    encoders['eastchester_encoder'] = eastchester_encoder

    # Edgewater Park
    edgewater_park_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'edgewater_park_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        edgewater_park_encoder.classes_ = json.load(infile)
    encoders['edgewater_park_encoder'] = edgewater_park_encoder

    # Flatlands
    flatlands_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'flatlands_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        flatlands_encoder.classes_ = json.load(infile)
    encoders['flatlands_encoder'] = flatlands_encoder

    # Fordham
    fordham_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'fordham_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        fordham_encoder.classes_ = json.load(infile)
    encoders['fordham_encoder'] = fordham_encoder

    # Fort Greene
    fort_greene_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'fort_greene_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        fort_greene_encoder.classes_ = json.load(infile)
    encoders['fort_greene_encoder'] = fort_greene_encoder

    # Fort Hamilton
    fort_hamilton_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'fort_hamilton_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        fort_hamilton_encoder.classes_ = json.load(infile)
    encoders['fort_hamilton_encoder'] = fort_hamilton_encoder

    # Georgetown
    georgetown_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'georgetown_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        georgetown_encoder.classes_ = json.load(infile)
    encoders['georgetown_encoder'] = georgetown_encoder

    # Gerritsen Beach
    gerritsen_beach_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'gerritsen_beach_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        gerritsen_beach_encoder.classes_ = json.load(infile)
    encoders['gerritsen_beach_encoder'] = gerritsen_beach_encoder

    # Gowanus
    gowanus_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'gowanus_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        gowanus_encoder.classes_ = json.load(infile)
    encoders['gowanus_encoder'] = gowanus_encoder

    # Greenpoint
    greenpoint_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'greenpoint_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        greenpoint_encoder.classes_ = json.load(infile)
    encoders['greenpoint_encoder'] = greenpoint_encoder

    # Hamilton Heights
    hamilton_heights_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'hamilton_heights_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        hamilton_heights_encoder.classes_ = json.load(infile)
    encoders['hamilton_heights_encoder'] = hamilton_heights_encoder

    # High  Bridge
    high_bridge_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'high_bridge_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        high_bridge_encoder.classes_ = json.load(infile)
    encoders['high_bridge_encoder'] = high_bridge_encoder

    # Inwood
    inwood_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'inwood_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        inwood_encoder.classes_ = json.load(infile)
    encoders['inwood_encoder'] = inwood_encoder

    # Kensington
    kensington_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'kensington_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        kensington_encoder.classes_ = json.load(infile)
    encoders['kensington_encoder'] = kensington_encoder

    # Kingsbridge
    kingsbridge_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'kingsbridge_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        kingsbridge_encoder.classes_ = json.load(infile)
    encoders['kingsbridge_encoder'] = kingsbridge_encoder

    # Lenox Hill
    lenox_hill_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'lenox_hill_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        lenox_hill_encoder.classes_ = json.load(infile)
    encoders['lenox_hill_encoder'] = lenox_hill_encoder

    # Lincoln Square
    lincoln_square_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'lincoln_square_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        lincoln_square_encoder.classes_ = json.load(infile)
    encoders['lincoln_square_encoder'] = lincoln_square_encoder

    # Manhattan Beach
    manhattan_beach_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'manhattan_beach_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        manhattan_beach_encoder.classes_ = json.load(infile)
    encoders['manhattan_beach_encoder'] = manhattan_beach_encoder

    # Manhattanville
    manhattanville_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'manhattanville_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        manhattanville_encoder.classes_ = json.load(infile)
    encoders['manhattanville_encoder'] = manhattanville_encoder

    # Marble Hill
    marble_hill_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'marble_hill_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        marble_hill_encoder.classes_ = json.load(infile)
    encoders['marble_hill_encoder'] = marble_hill_encoder

    # Melrose
    melrose_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'melrose_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        melrose_encoder.classes_ = json.load(infile)
    encoders['melrose_encoder'] = melrose_encoder

    # Midtown
    midtown_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'midtown_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        midtown_encoder.classes_ = json.load(infile)
    encoders['midtown_encoder'] = midtown_encoder

    # Morris Park
    morris_park_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'morris_park_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        morris_park_encoder.classes_ = json.load(infile)
    encoders['morris_park_encoder'] = morris_park_encoder

    # Morrisania
    morrisania_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'morrisania_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        morrisania_encoder.classes_ = json.load(infile)
    encoders['morrisania_encoder'] = morrisania_encoder

    # Mott Haven
    mott_haven_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'mott_haven_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        mott_haven_encoder.classes_ = json.load(infile)
    encoders['mott_haven_encoder'] = mott_haven_encoder

    # Navy Yard
    navy_yard_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'navy_yard_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        navy_yard_encoder.classes_ = json.load(infile)
    encoders['navy_yard_encoder'] = navy_yard_encoder

    # North Riverdale
    north_riverdale_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'north_riverdale_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        north_riverdale_encoder.classes_ = json.load(infile)
    encoders['north_riverdale_encoder'] = north_riverdale_encoder

    # North Side
    north_side_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'north_side_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        north_side_encoder.classes_ = json.load(infile)
    encoders['north_side_encoder'] = north_side_encoder

    # Norwood
    norwood_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'norwood_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        norwood_encoder.classes_ = json.load(infile)
    encoders['norwood_encoder'] = norwood_encoder

    # Park Slope
    park_slope_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'park_slope_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        park_slope_encoder.classes_ = json.load(infile)
    encoders['park_slope_encoder'] = park_slope_encoder

    # Parkchester
    parkchester_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'parkchester_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        parkchester_encoder.classes_ = json.load(infile)
    encoders['parkchester_encoder'] = parkchester_encoder

    # Prospect Heights
    prospect_heights_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'prospect_heights_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        prospect_heights_encoder.classes_ = json.load(infile)
    encoders['prospect_heights_encoder'] = prospect_heights_encoder

    # Prospect Lefferts Gardens
    prospect_lefferts_gardens_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'prospect_lefferts_gardens_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        prospect_lefferts_gardens_encoder.classes_ = json.load(infile)
    encoders['prospect_lefferts_gardens_encoder'] = prospect_lefferts_gardens_encoder

    # Prospect Park South
    prospect_park_south_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'prospect_park_south_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        prospect_park_south_encoder.classes_ = json.load(infile)
    encoders['prospect_park_south_encoder'] = prospect_park_south_encoder

    # Red Hook
    red_hook_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'red_hook_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        red_hook_encoder.classes_ = json.load(infile)
    encoders['red_hook_encoder'] = red_hook_encoder

    # Schuylerville
    schuylerville_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'schuylerville_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        schuylerville_encoder.classes_ = json.load(infile)
    encoders['schuylerville_encoder'] = schuylerville_encoder

    # Sheepshead Bay
    sheepshead_bay_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'sheepshead_bay_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        sheepshead_bay_encoder.classes_ = json.load(infile)
    encoders['sheepshead_bay_encoder'] = sheepshead_bay_encoder

    # Soundview
    soundview_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'soundview_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        soundview_encoder.classes_ = json.load(infile)
    encoders['soundview_encoder'] = soundview_encoder

    # South Side
    south_side_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'south_side_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        south_side_encoder.classes_ = json.load(infile)
    encoders['south_side_encoder'] = south_side_encoder

    # Spring Creek
    spring_creek_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'spring_creek_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        spring_creek_encoder.classes_ = json.load(infile)
    encoders['spring_creek_encoder'] = spring_creek_encoder

    # Throgs Neck
    throgs_neck_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'throgs_neck_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        throgs_neck_encoder.classes_ = json.load(infile)
    encoders['throgs_neck_encoder'] = throgs_neck_encoder

    # Upper East Side
    upper_east_side_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'upper_east_side_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        upper_east_side_encoder.classes_ = json.load(infile)
    encoders['upper_east_side_encoder'] = upper_east_side_encoder

    # Upper West Side
    upper_west_side_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'upper_west_side_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        upper_west_side_encoder.classes_ = json.load(infile)
    encoders['upper_west_side_encoder'] = upper_west_side_encoder

    # Van Nest
    van_nest_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'van_nest_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        van_nest_encoder.classes_ = json.load(infile)
    encoders['van_nest_encoder'] = van_nest_encoder

    # Washington Heights
    washington_heights_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'washington_heights_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        washington_heights_encoder.classes_ = json.load(infile)
    encoders['washington_heights_encoder'] = washington_heights_encoder

    # Westchester Square
    westchester_square_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'westchester_square_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        westchester_square_encoder.classes_ = json.load(infile)
    encoders['westchester_square_encoder'] = westchester_square_encoder

    # Williamsburg
    williamsburg_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'williamsburg_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        williamsburg_encoder.classes_ = json.load(infile)
    encoders['williamsburg_encoder'] = williamsburg_encoder

    # Windsor Terrace
    windsor_terrace_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'windsor_terrace_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        windsor_terrace_encoder.classes_ = json.load(infile)
    encoders['windsor_terrace_encoder'] = windsor_terrace_encoder

    # Woodlawn
    woodlawn_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'woodlawn_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        woodlawn_encoder.classes_ = json.load(infile)
    encoders['woodlawn_encoder'] = woodlawn_encoder

    # Yorkville
    yorkville_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'yorkville_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        yorkville_encoder.classes_ = json.load(infile)
    encoders['yorkville_encoder'] = yorkville_encoder

    # Tip Count
    tip_count_encoder = LabelBinarizer()
    tip_count_encoder.classes_ = list(range(4))

    with open(os.path.join('encoders', 'tip_count_bins.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        tip_count_bins = json.load(infile)
    encoders['tip_count_bins'] = tip_count_bins
    encoders['tip_count_encoder'] = tip_count_encoder

    # Like Count
    like_count_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'like_count_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        like_count_encoder.classes_ = json.load(infile)
    encoders['like_count_encoder'] = like_count_encoder

    # Target Field: Rating

    return encoders


def process_data(df, encoders, process_target=True):
    """Processes an input DataFrame into a format
    sutable for model prediction.

    This function loads the encoder specifications created in
    `build_encoders`.

    # Arguments
        df: a DataFrame containing the source data
        encoders: a dict of encoders to process the data.
        process_target: boolean to determine if the target should be encoded.

    # Returns
        A tuple: A list containing all the processed fields to be fed
        into the model, and the processed target field.
    """

    # Unnamed: 0
    unnamed_0_enc = pd.cut(
        df['Unnamed: 0'].values, encoders['unnamed_0_bins'], labels=False, include_lowest=True)
    unnamed_0_enc = encoders['unnamed_0_encoder'].transform(unnamed_0_enc)

    # American Restaurant
    american_restaurant_enc = df['American Restaurant'].values
    american_restaurant_enc = encoders['american_restaurant_encoder'].transform(
        american_restaurant_enc)

    # Asian Restaurant
    asian_restaurant_enc = df['Asian Restaurant'].values
    asian_restaurant_enc = encoders['asian_restaurant_encoder'].transform(
        asian_restaurant_enc)

    # Australian Restaurant
    australian_restaurant_enc = df['Australian Restaurant'].values
    australian_restaurant_enc = encoders['australian_restaurant_encoder'].transform(
        australian_restaurant_enc)

    # BBQ Joint
    bbq_joint_enc = df['BBQ Joint'].values
    bbq_joint_enc = encoders['bbq_joint_encoder'].transform(bbq_joint_enc)

    # Bakery
    bakery_enc = df['Bakery'].values
    bakery_enc = encoders['bakery_encoder'].transform(bakery_enc)

    # Bar
    bar_enc = df['Bar'].values
    bar_enc = encoders['bar_encoder'].transform(bar_enc)

    # Beer Garden
    beer_garden_enc = df['Beer Garden'].values
    beer_garden_enc = encoders['beer_garden_encoder'].transform(
        beer_garden_enc)

    # Breakfast Spot
    breakfast_spot_enc = df['Breakfast Spot'].values
    breakfast_spot_enc = encoders['breakfast_spot_encoder'].transform(
        breakfast_spot_enc)

    # Brewery
    brewery_enc = df['Brewery'].values
    brewery_enc = encoders['brewery_encoder'].transform(brewery_enc)

    # Burger Joint
    burger_joint_enc = df['Burger Joint'].values
    burger_joint_enc = encoders['burger_joint_encoder'].transform(
        burger_joint_enc)

    # Café
    café_enc = df['Café'].values
    café_enc = encoders['café_encoder'].transform(café_enc)

    # Cocktail Bar
    cocktail_bar_enc = df['Cocktail Bar'].values
    cocktail_bar_enc = encoders['cocktail_bar_encoder'].transform(
        cocktail_bar_enc)

    # Coffee Shop
    coffee_shop_enc = df['Coffee Shop'].values
    coffee_shop_enc = encoders['coffee_shop_encoder'].transform(
        coffee_shop_enc)

    # Deli / Bodega
    deli_bodega_enc = df['Deli / Bodega'].values
    deli_bodega_enc = encoders['deli_bodega_encoder'].transform(
        deli_bodega_enc)

    # Diner
    diner_enc = df['Diner'].values
    diner_enc = encoders['diner_encoder'].transform(diner_enc)

    # Dive Bar
    dive_bar_enc = df['Dive Bar'].values
    dive_bar_enc = encoders['dive_bar_encoder'].transform(dive_bar_enc)

    # Eastern European Restaurant
    eastern_european_restaurant_enc = df['Eastern European Restaurant'].values
    eastern_european_restaurant_enc = encoders['eastern_european_restaurant_encoder'].transform(
        eastern_european_restaurant_enc)

    # Fast Food Restaurant
    fast_food_restaurant_enc = df['Fast Food Restaurant'].values
    fast_food_restaurant_enc = encoders['fast_food_restaurant_encoder'].transform(
        fast_food_restaurant_enc)

    # Food & Drink Shop
    food_drink_shop_enc = df['Food & Drink Shop'].values
    food_drink_shop_enc = encoders['food_drink_shop_encoder'].transform(
        food_drink_shop_enc)

    # Food Truck
    food_truck_enc = df['Food Truck'].values
    food_truck_enc = encoders['food_truck_encoder'].transform(food_truck_enc)

    # French Restaurant
    french_restaurant_enc = df['French Restaurant'].values
    french_restaurant_enc = encoders['french_restaurant_encoder'].transform(
        french_restaurant_enc)

    # Fried Chicken Joint
    fried_chicken_joint_enc = df['Fried Chicken Joint'].values
    fried_chicken_joint_enc = encoders['fried_chicken_joint_encoder'].transform(
        fried_chicken_joint_enc)

    # Gastropub
    gastropub_enc = df['Gastropub'].values
    gastropub_enc = encoders['gastropub_encoder'].transform(gastropub_enc)

    # Greek Restaurant
    greek_restaurant_enc = df['Greek Restaurant'].values
    greek_restaurant_enc = encoders['greek_restaurant_encoder'].transform(
        greek_restaurant_enc)

    # Hookah Bar
    hookah_bar_enc = df['Hookah Bar'].values
    hookah_bar_enc = encoders['hookah_bar_encoder'].transform(hookah_bar_enc)

    # Hot Dog Joint
    hot_dog_joint_enc = df['Hot Dog Joint'].values
    hot_dog_joint_enc = encoders['hot_dog_joint_encoder'].transform(
        hot_dog_joint_enc)

    # Hotel Bar
    hotel_bar_enc = df['Hotel Bar'].values
    hotel_bar_enc = encoders['hotel_bar_encoder'].transform(hotel_bar_enc)

    # Irish Pub
    irish_pub_enc = df['Irish Pub'].values
    irish_pub_enc = encoders['irish_pub_encoder'].transform(irish_pub_enc)

    # Italian Restaurant
    italian_restaurant_enc = df['Italian Restaurant'].values
    italian_restaurant_enc = encoders['italian_restaurant_encoder'].transform(
        italian_restaurant_enc)

    # Juice Bar
    juice_bar_enc = df['Juice Bar'].values
    juice_bar_enc = encoders['juice_bar_encoder'].transform(juice_bar_enc)

    # Lounge
    lounge_enc = df['Lounge'].values
    lounge_enc = encoders['lounge_encoder'].transform(lounge_enc)

    # Mediterranean Restaurant
    mediterranean_restaurant_enc = df['Mediterranean Restaurant'].values
    mediterranean_restaurant_enc = encoders['mediterranean_restaurant_encoder'].transform(
        mediterranean_restaurant_enc)

    # New American Restaurant
    new_american_restaurant_enc = df['New American Restaurant'].values
    new_american_restaurant_enc = encoders['new_american_restaurant_encoder'].transform(
        new_american_restaurant_enc)

    # Piano Bar
    piano_bar_enc = df['Piano Bar'].values
    piano_bar_enc = encoders['piano_bar_encoder'].transform(piano_bar_enc)

    # Pizza Place
    pizza_place_enc = df['Pizza Place'].values
    pizza_place_enc = encoders['pizza_place_encoder'].transform(
        pizza_place_enc)

    # Pub
    pub_enc = df['Pub'].values
    pub_enc = encoders['pub_encoder'].transform(pub_enc)

    # Restaurant
    restaurant_enc = df['Restaurant'].values
    restaurant_enc = encoders['restaurant_encoder'].transform(restaurant_enc)

    # Rock Club
    rock_club_enc = df['Rock Club'].values
    rock_club_enc = encoders['rock_club_encoder'].transform(rock_club_enc)

    # Sandwich Place
    sandwich_place_enc = df['Sandwich Place'].values
    sandwich_place_enc = encoders['sandwich_place_encoder'].transform(
        sandwich_place_enc)

    # Seafood Restaurant
    seafood_restaurant_enc = df['Seafood Restaurant'].values
    seafood_restaurant_enc = encoders['seafood_restaurant_encoder'].transform(
        seafood_restaurant_enc)

    # Southern / Soul Food Restaurant
    southern_soul_food_restaurant_enc = df['Southern / Soul Food Restaurant'].values
    southern_soul_food_restaurant_enc = encoders['southern_soul_food_restaurant_encoder'].transform(
        southern_soul_food_restaurant_enc)

    # Sports Bar
    sports_bar_enc = df['Sports Bar'].values
    sports_bar_enc = encoders['sports_bar_encoder'].transform(sports_bar_enc)

    # Steakhouse
    steakhouse_enc = df['Steakhouse'].values
    steakhouse_enc = encoders['steakhouse_encoder'].transform(steakhouse_enc)

    # Theme Restaurant
    theme_restaurant_enc = df['Theme Restaurant'].values
    theme_restaurant_enc = encoders['theme_restaurant_encoder'].transform(
        theme_restaurant_enc)

    # Vegetarian / Vegan Restaurant
    vegetarian_vegan_restaurant_enc = df['Vegetarian / Vegan Restaurant'].values
    vegetarian_vegan_restaurant_enc = encoders['vegetarian_vegan_restaurant_encoder'].transform(
        vegetarian_vegan_restaurant_enc)

    # Whisky Bar
    whisky_bar_enc = df['Whisky Bar'].values
    whisky_bar_enc = encoders['whisky_bar_encoder'].transform(whisky_bar_enc)

    # Wings Joint
    wings_joint_enc = df['Wings Joint'].values
    wings_joint_enc = encoders['wings_joint_encoder'].transform(
        wings_joint_enc)

    # Bath Beach
    bath_beach_enc = df['Bath Beach'].values
    bath_beach_enc = encoders['bath_beach_encoder'].transform(bath_beach_enc)

    # Bay Ridge
    bay_ridge_enc = df['Bay Ridge'].values
    bay_ridge_enc = encoders['bay_ridge_encoder'].transform(bay_ridge_enc)

    # Baychester
    baychester_enc = df['Baychester'].values
    baychester_enc = encoders['baychester_encoder'].transform(baychester_enc)

    # Bedford Park
    bedford_park_enc = df['Bedford Park'].values
    bedford_park_enc = encoders['bedford_park_encoder'].transform(
        bedford_park_enc)

    # Bedford Stuyvesant
    bedford_stuyvesant_enc = df['Bedford Stuyvesant'].values
    bedford_stuyvesant_enc = encoders['bedford_stuyvesant_encoder'].transform(
        bedford_stuyvesant_enc)

    # Belmont
    belmont_enc = df['Belmont'].values
    belmont_enc = encoders['belmont_encoder'].transform(belmont_enc)

    # Boerum Hill
    boerum_hill_enc = df['Boerum Hill'].values
    boerum_hill_enc = encoders['boerum_hill_encoder'].transform(
        boerum_hill_enc)

    # Brighton Beach
    brighton_beach_enc = df['Brighton Beach'].values
    brighton_beach_enc = encoders['brighton_beach_encoder'].transform(
        brighton_beach_enc)

    # Brooklyn Heights
    brooklyn_heights_enc = df['Brooklyn Heights'].values
    brooklyn_heights_enc = encoders['brooklyn_heights_encoder'].transform(
        brooklyn_heights_enc)

    # Bushwick
    bushwick_enc = df['Bushwick'].values
    bushwick_enc = encoders['bushwick_encoder'].transform(bushwick_enc)

    # Carroll Gardens
    carroll_gardens_enc = df['Carroll Gardens'].values
    carroll_gardens_enc = encoders['carroll_gardens_encoder'].transform(
        carroll_gardens_enc)

    # Central Harlem
    central_harlem_enc = df['Central Harlem'].values
    central_harlem_enc = encoders['central_harlem_encoder'].transform(
        central_harlem_enc)

    # Chinatown
    chinatown_enc = df['Chinatown'].values
    chinatown_enc = encoders['chinatown_encoder'].transform(chinatown_enc)

    # City Island
    city_island_enc = df['City Island'].values
    city_island_enc = encoders['city_island_encoder'].transform(
        city_island_enc)

    # City Line
    city_line_enc = df['City Line'].values
    city_line_enc = encoders['city_line_encoder'].transform(city_line_enc)

    # Clinton
    clinton_enc = df['Clinton'].values
    clinton_enc = encoders['clinton_encoder'].transform(clinton_enc)

    # Clinton Hill
    clinton_hill_enc = df['Clinton Hill'].values
    clinton_hill_enc = encoders['clinton_hill_encoder'].transform(
        clinton_hill_enc)

    # Co-op City
    co_op_city_enc = df['Co-op City'].values
    co_op_city_enc = encoders['co_op_city_encoder'].transform(co_op_city_enc)

    # Cobble Hill
    cobble_hill_enc = df['Cobble Hill'].values
    cobble_hill_enc = encoders['cobble_hill_encoder'].transform(
        cobble_hill_enc)

    # Crown Heights
    crown_heights_enc = df['Crown Heights'].values
    crown_heights_enc = encoders['crown_heights_encoder'].transform(
        crown_heights_enc)

    # Downtown
    downtown_enc = df['Downtown'].values
    downtown_enc = encoders['downtown_encoder'].transform(downtown_enc)

    # Dyker Heights
    dyker_heights_enc = df['Dyker Heights'].values
    dyker_heights_enc = encoders['dyker_heights_encoder'].transform(
        dyker_heights_enc)

    # East Harlem
    east_harlem_enc = df['East Harlem'].values
    east_harlem_enc = encoders['east_harlem_encoder'].transform(
        east_harlem_enc)

    # East Tremont
    east_tremont_enc = df['East Tremont'].values
    east_tremont_enc = encoders['east_tremont_encoder'].transform(
        east_tremont_enc)

    # East Williamsburg
    east_williamsburg_enc = df['East Williamsburg'].values
    east_williamsburg_enc = encoders['east_williamsburg_encoder'].transform(
        east_williamsburg_enc)

    # Eastchester
    eastchester_enc = df['Eastchester'].values
    eastchester_enc = encoders['eastchester_encoder'].transform(
        eastchester_enc)

    # Edgewater Park
    edgewater_park_enc = df['Edgewater Park'].values
    edgewater_park_enc = encoders['edgewater_park_encoder'].transform(
        edgewater_park_enc)

    # Flatlands
    flatlands_enc = df['Flatlands'].values
    flatlands_enc = encoders['flatlands_encoder'].transform(flatlands_enc)

    # Fordham
    fordham_enc = df['Fordham'].values
    fordham_enc = encoders['fordham_encoder'].transform(fordham_enc)

    # Fort Greene
    fort_greene_enc = df['Fort Greene'].values
    fort_greene_enc = encoders['fort_greene_encoder'].transform(
        fort_greene_enc)

    # Fort Hamilton
    fort_hamilton_enc = df['Fort Hamilton'].values
    fort_hamilton_enc = encoders['fort_hamilton_encoder'].transform(
        fort_hamilton_enc)

    # Georgetown
    georgetown_enc = df['Georgetown'].values
    georgetown_enc = encoders['georgetown_encoder'].transform(georgetown_enc)

    # Gerritsen Beach
    gerritsen_beach_enc = df['Gerritsen Beach'].values
    gerritsen_beach_enc = encoders['gerritsen_beach_encoder'].transform(
        gerritsen_beach_enc)

    # Gowanus
    gowanus_enc = df['Gowanus'].values
    gowanus_enc = encoders['gowanus_encoder'].transform(gowanus_enc)

    # Greenpoint
    greenpoint_enc = df['Greenpoint'].values
    greenpoint_enc = encoders['greenpoint_encoder'].transform(greenpoint_enc)

    # Hamilton Heights
    hamilton_heights_enc = df['Hamilton Heights'].values
    hamilton_heights_enc = encoders['hamilton_heights_encoder'].transform(
        hamilton_heights_enc)

    # High  Bridge
    high_bridge_enc = df['High  Bridge'].values
    high_bridge_enc = encoders['high_bridge_encoder'].transform(
        high_bridge_enc)

    # Inwood
    inwood_enc = df['Inwood'].values
    inwood_enc = encoders['inwood_encoder'].transform(inwood_enc)

    # Kensington
    kensington_enc = df['Kensington'].values
    kensington_enc = encoders['kensington_encoder'].transform(kensington_enc)

    # Kingsbridge
    kingsbridge_enc = df['Kingsbridge'].values
    kingsbridge_enc = encoders['kingsbridge_encoder'].transform(
        kingsbridge_enc)

    # Lenox Hill
    lenox_hill_enc = df['Lenox Hill'].values
    lenox_hill_enc = encoders['lenox_hill_encoder'].transform(lenox_hill_enc)

    # Lincoln Square
    lincoln_square_enc = df['Lincoln Square'].values
    lincoln_square_enc = encoders['lincoln_square_encoder'].transform(
        lincoln_square_enc)

    # Manhattan Beach
    manhattan_beach_enc = df['Manhattan Beach'].values
    manhattan_beach_enc = encoders['manhattan_beach_encoder'].transform(
        manhattan_beach_enc)

    # Manhattanville
    manhattanville_enc = df['Manhattanville'].values
    manhattanville_enc = encoders['manhattanville_encoder'].transform(
        manhattanville_enc)

    # Marble Hill
    marble_hill_enc = df['Marble Hill'].values
    marble_hill_enc = encoders['marble_hill_encoder'].transform(
        marble_hill_enc)

    # Melrose
    melrose_enc = df['Melrose'].values
    melrose_enc = encoders['melrose_encoder'].transform(melrose_enc)

    # Midtown
    midtown_enc = df['Midtown'].values
    midtown_enc = encoders['midtown_encoder'].transform(midtown_enc)

    # Morris Park
    morris_park_enc = df['Morris Park'].values
    morris_park_enc = encoders['morris_park_encoder'].transform(
        morris_park_enc)

    # Morrisania
    morrisania_enc = df['Morrisania'].values
    morrisania_enc = encoders['morrisania_encoder'].transform(morrisania_enc)

    # Mott Haven
    mott_haven_enc = df['Mott Haven'].values
    mott_haven_enc = encoders['mott_haven_encoder'].transform(mott_haven_enc)

    # Navy Yard
    navy_yard_enc = df['Navy Yard'].values
    navy_yard_enc = encoders['navy_yard_encoder'].transform(navy_yard_enc)

    # North Riverdale
    north_riverdale_enc = df['North Riverdale'].values
    north_riverdale_enc = encoders['north_riverdale_encoder'].transform(
        north_riverdale_enc)

    # North Side
    north_side_enc = df['North Side'].values
    north_side_enc = encoders['north_side_encoder'].transform(north_side_enc)

    # Norwood
    norwood_enc = df['Norwood'].values
    norwood_enc = encoders['norwood_encoder'].transform(norwood_enc)

    # Park Slope
    park_slope_enc = df['Park Slope'].values
    park_slope_enc = encoders['park_slope_encoder'].transform(park_slope_enc)

    # Parkchester
    parkchester_enc = df['Parkchester'].values
    parkchester_enc = encoders['parkchester_encoder'].transform(
        parkchester_enc)

    # Prospect Heights
    prospect_heights_enc = df['Prospect Heights'].values
    prospect_heights_enc = encoders['prospect_heights_encoder'].transform(
        prospect_heights_enc)

    # Prospect Lefferts Gardens
    prospect_lefferts_gardens_enc = df['Prospect Lefferts Gardens'].values
    prospect_lefferts_gardens_enc = encoders['prospect_lefferts_gardens_encoder'].transform(
        prospect_lefferts_gardens_enc)

    # Prospect Park South
    prospect_park_south_enc = df['Prospect Park South'].values
    prospect_park_south_enc = encoders['prospect_park_south_encoder'].transform(
        prospect_park_south_enc)

    # Red Hook
    red_hook_enc = df['Red Hook'].values
    red_hook_enc = encoders['red_hook_encoder'].transform(red_hook_enc)

    # Schuylerville
    schuylerville_enc = df['Schuylerville'].values
    schuylerville_enc = encoders['schuylerville_encoder'].transform(
        schuylerville_enc)

    # Sheepshead Bay
    sheepshead_bay_enc = df['Sheepshead Bay'].values
    sheepshead_bay_enc = encoders['sheepshead_bay_encoder'].transform(
        sheepshead_bay_enc)

    # Soundview
    soundview_enc = df['Soundview'].values
    soundview_enc = encoders['soundview_encoder'].transform(soundview_enc)

    # South Side
    south_side_enc = df['South Side'].values
    south_side_enc = encoders['south_side_encoder'].transform(south_side_enc)

    # Spring Creek
    spring_creek_enc = df['Spring Creek'].values
    spring_creek_enc = encoders['spring_creek_encoder'].transform(
        spring_creek_enc)

    # Throgs Neck
    throgs_neck_enc = df['Throgs Neck'].values
    throgs_neck_enc = encoders['throgs_neck_encoder'].transform(
        throgs_neck_enc)

    # Upper East Side
    upper_east_side_enc = df['Upper East Side'].values
    upper_east_side_enc = encoders['upper_east_side_encoder'].transform(
        upper_east_side_enc)

    # Upper West Side
    upper_west_side_enc = df['Upper West Side'].values
    upper_west_side_enc = encoders['upper_west_side_encoder'].transform(
        upper_west_side_enc)

    # Van Nest
    van_nest_enc = df['Van Nest'].values
    van_nest_enc = encoders['van_nest_encoder'].transform(van_nest_enc)

    # Washington Heights
    washington_heights_enc = df['Washington Heights'].values
    washington_heights_enc = encoders['washington_heights_encoder'].transform(
        washington_heights_enc)

    # Westchester Square
    westchester_square_enc = df['Westchester Square'].values
    westchester_square_enc = encoders['westchester_square_encoder'].transform(
        westchester_square_enc)

    # Williamsburg
    williamsburg_enc = df['Williamsburg'].values
    williamsburg_enc = encoders['williamsburg_encoder'].transform(
        williamsburg_enc)

    # Windsor Terrace
    windsor_terrace_enc = df['Windsor Terrace'].values
    windsor_terrace_enc = encoders['windsor_terrace_encoder'].transform(
        windsor_terrace_enc)

    # Woodlawn
    woodlawn_enc = df['Woodlawn'].values
    woodlawn_enc = encoders['woodlawn_encoder'].transform(woodlawn_enc)

    # Yorkville
    yorkville_enc = df['Yorkville'].values
    yorkville_enc = encoders['yorkville_encoder'].transform(yorkville_enc)

    # Tip Count
    tip_count_enc = pd.cut(
        df['Tip Count'].values, encoders['tip_count_bins'], labels=False, include_lowest=True)
    tip_count_enc = encoders['tip_count_encoder'].transform(tip_count_enc)

    # Like Count
    like_count_enc = df['Like Count'].values
    like_count_enc = encoders['like_count_encoder'].transform(like_count_enc)

    data_enc = [unnamed_0_enc,
                american_restaurant_enc,
                asian_restaurant_enc,
                australian_restaurant_enc,
                bbq_joint_enc,
                bakery_enc,
                bar_enc,
                beer_garden_enc,
                breakfast_spot_enc,
                brewery_enc,
                burger_joint_enc,
                café_enc,
                cocktail_bar_enc,
                coffee_shop_enc,
                deli_bodega_enc,
                diner_enc,
                dive_bar_enc,
                eastern_european_restaurant_enc,
                fast_food_restaurant_enc,
                food_drink_shop_enc,
                food_truck_enc,
                french_restaurant_enc,
                fried_chicken_joint_enc,
                gastropub_enc,
                greek_restaurant_enc,
                hookah_bar_enc,
                hot_dog_joint_enc,
                hotel_bar_enc,
                irish_pub_enc,
                italian_restaurant_enc,
                juice_bar_enc,
                lounge_enc,
                mediterranean_restaurant_enc,
                new_american_restaurant_enc,
                piano_bar_enc,
                pizza_place_enc,
                pub_enc,
                restaurant_enc,
                rock_club_enc,
                sandwich_place_enc,
                seafood_restaurant_enc,
                southern_soul_food_restaurant_enc,
                sports_bar_enc,
                steakhouse_enc,
                theme_restaurant_enc,
                vegetarian_vegan_restaurant_enc,
                whisky_bar_enc,
                wings_joint_enc,
                bath_beach_enc,
                bay_ridge_enc,
                baychester_enc,
                bedford_park_enc,
                bedford_stuyvesant_enc,
                belmont_enc,
                boerum_hill_enc,
                brighton_beach_enc,
                brooklyn_heights_enc,
                bushwick_enc,
                carroll_gardens_enc,
                central_harlem_enc,
                chinatown_enc,
                city_island_enc,
                city_line_enc,
                clinton_enc,
                clinton_hill_enc,
                co_op_city_enc,
                cobble_hill_enc,
                crown_heights_enc,
                downtown_enc,
                dyker_heights_enc,
                east_harlem_enc,
                east_tremont_enc,
                east_williamsburg_enc,
                eastchester_enc,
                edgewater_park_enc,
                flatlands_enc,
                fordham_enc,
                fort_greene_enc,
                fort_hamilton_enc,
                georgetown_enc,
                gerritsen_beach_enc,
                gowanus_enc,
                greenpoint_enc,
                hamilton_heights_enc,
                high_bridge_enc,
                inwood_enc,
                kensington_enc,
                kingsbridge_enc,
                lenox_hill_enc,
                lincoln_square_enc,
                manhattan_beach_enc,
                manhattanville_enc,
                marble_hill_enc,
                melrose_enc,
                midtown_enc,
                morris_park_enc,
                morrisania_enc,
                mott_haven_enc,
                navy_yard_enc,
                north_riverdale_enc,
                north_side_enc,
                norwood_enc,
                park_slope_enc,
                parkchester_enc,
                prospect_heights_enc,
                prospect_lefferts_gardens_enc,
                prospect_park_south_enc,
                red_hook_enc,
                schuylerville_enc,
                sheepshead_bay_enc,
                soundview_enc,
                south_side_enc,
                spring_creek_enc,
                throgs_neck_enc,
                upper_east_side_enc,
                upper_west_side_enc,
                van_nest_enc,
                washington_heights_enc,
                westchester_square_enc,
                williamsburg_enc,
                windsor_terrace_enc,
                woodlawn_enc,
                yorkville_enc,
                tip_count_enc,
                like_count_enc
                ]

    if process_target:
        # Target Field: Rating
        rating_enc = df['Rating'].values

        return (data_enc, rating_enc)

    return data_enc


def model_predict(df, model, encoders):
    """Generates predictions for a trained model.

    # Arguments
        df: A pandas DataFrame containing the source data.
        model: A compiled model.
        encoders: a dict of encoders to process the data.

    # Returns
        A numpy array of predictions.
    """

    data_enc = process_data(df, encoders, process_target=False)

    data_enc = xgb.DMatrix(np.hstack(data_enc))

    headers = ['rating']
    predictions = pd.DataFrame(model.predict(data_enc), columns=headers)

    return predictions


def model_train(df, encoders, args, model=None):
    """Trains a model, and saves the data locally.

    # Arguments
        df: A pandas DataFrame containing the source data.
        encoders: a dict of encoders to process the data.
        args: a dict of arguments passed through the command line
        model: A compiled model (for TensorFlow, None otherwise).
    """

    X, y_enc = process_data(df, encoders)
    X = np.hstack(X)
    y = df['Rating'].values

    split = ShuffleSplit(n_splits=1, train_size=args.split,
                         test_size=None, random_state=123)

    for train_indices, val_indices in split.split(np.zeros(y.shape[0]), y):
        train = xgb.DMatrix(X[train_indices, ], y[train_indices, ])
        val = xgb.DMatrix(X[val_indices, ], y[val_indices, ])

    params = {
        'eta': 0.1,
        'max_depth': 3,
        'gamma': 1,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'max_bin': 512,
        'objective': 'reg:linear',
        'tree_method': 'hist',
        'silent': 1
    }

    f = open(os.path.join('metadata', 'results.csv'), 'w')
    w = csv.writer(f)
    w.writerow(['epoch', 'time_completed'] + ['mse', 'mae', 'r_2'])

    y_true = y_enc[val_indices, ]
    for epoch in range(args.epochs):
        model = xgb.train(params, train, 1,
                          xgb_model=model if epoch > 0 else None)
        y_pred = model.predict(val)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r_2 = r2_score(y_true, y_pred)

        metrics = [mse, mae, r_2]
        time_completed = "{:%Y-%m-%d %H:%M:%S}".format(datetime.utcnow())
        w.writerow([epoch+1, time_completed] + metrics)

        if args.context == 'automl-gs':
            sys.stdout.flush()
            print("\nEPOCH_END")

    f.close()
    model.save_model('model.bin')
