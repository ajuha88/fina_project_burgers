import argparse
import pandas as pd
from pipeline import *
import xgboost as xgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A script which utilizes a model trained to predict rating."
                    "Script created using automl-gs (https://github.com/minimaxir/automl-gs)")
    parser.add_argument('-d', '--data',  help="Input dataset (must be a .csv)")
    parser.add_argument(
        '-m', '--mode',  help='Mode (either "train" or "predict")')
    parser.add_argument(
        '-s', '--split',  help="Train/Validation Split (if training)",
        default=0.7)
    parser.add_argument(
        '-e', '--epochs',  help="# of Epochs (if training)",
        default=10)
    parser.add_argument(
        '-c', '--context',  help="Context for running script (used during automl-gs training)",
        default='standalone')
    parser.add_argument(
        '-t', '--type',  help="Format for predictions (either csv or json)",
        default='csv')
    args = parser.parse_args()

    cols = ["Unnamed: 0",
            "American Restaurant",
            "Asian Restaurant",
            "Australian Restaurant",
            "BBQ Joint",
            "Bakery",
            "Bar",
            "Beer Garden",
            "Breakfast Spot",
            "Brewery",
            "Burger Joint",
            "Café",
            "Cocktail Bar",
            "Coffee Shop",
            "Deli / Bodega",
            "Diner",
            "Dive Bar",
            "Eastern European Restaurant",
            "Fast Food Restaurant",
            "Food & Drink Shop",
            "Food Truck",
            "French Restaurant",
            "Fried Chicken Joint",
            "Gastropub",
            "Greek Restaurant",
            "Hookah Bar",
            "Hot Dog Joint",
            "Hotel Bar",
            "Irish Pub",
            "Italian Restaurant",
            "Juice Bar",
            "Lounge",
            "Mediterranean Restaurant",
            "New American Restaurant",
            "Piano Bar",
            "Pizza Place",
            "Pub",
            "Restaurant",
            "Rock Club",
            "Sandwich Place",
            "Seafood Restaurant",
            "Southern / Soul Food Restaurant",
            "Sports Bar",
            "Steakhouse",
            "Theme Restaurant",
            "Vegetarian / Vegan Restaurant",
            "Whisky Bar",
            "Wings Joint",
            "Bath Beach",
            "Bay Ridge",
            "Baychester",
            "Bedford Park",
            "Bedford Stuyvesant",
            "Belmont",
            "Boerum Hill",
            "Brighton Beach",
            "Brooklyn Heights",
            "Bushwick",
            "Carroll Gardens",
            "Central Harlem",
            "Chinatown",
            "City Island",
            "City Line",
            "Clinton",
            "Clinton Hill",
            "Co-op City",
            "Cobble Hill",
            "Crown Heights",
            "Downtown",
            "Dyker Heights",
            "East Harlem",
            "East Tremont",
            "East Williamsburg",
            "Eastchester",
            "Edgewater Park",
            "Flatlands",
            "Fordham",
            "Fort Greene",
            "Fort Hamilton",
            "Georgetown",
            "Gerritsen Beach",
            "Gowanus",
            "Greenpoint",
            "Hamilton Heights",
            "High  Bridge",
            "Inwood",
            "Kensington",
            "Kingsbridge",
            "Lenox Hill",
            "Lincoln Square",
            "Manhattan Beach",
            "Manhattanville",
            "Marble Hill",
            "Melrose",
            "Midtown",
            "Morris Park",
            "Morrisania",
            "Mott Haven",
            "Navy Yard",
            "North Riverdale",
            "North Side",
            "Norwood",
            "Park Slope",
            "Parkchester",
            "Prospect Heights",
            "Prospect Lefferts Gardens",
            "Prospect Park South",
            "Red Hook",
            "Schuylerville",
            "Sheepshead Bay",
            "Soundview",
            "South Side",
            "Spring Creek",
            "Throgs Neck",
            "Upper East Side",
            "Upper West Side",
            "Van Nest",
            "Washington Heights",
            "Westchester Square",
            "Williamsburg",
            "Windsor Terrace",
            "Woodlawn",
            "Yorkville",
            "Tip Count",
            "Rating",
            "Like Count"
            ]
    dtypes = {'Unnamed: 0': 'float64', 'American Restaurant': 'str', 'Asian Restaurant': 'str', 'Australian Restaurant': 'str', 'BBQ Joint': 'str', 'Bakery': 'str', 'Bar': 'str', 'Beer Garden': 'str', 'Breakfast Spot': 'str', 'Brewery': 'str', 'Burger Joint': 'str', 'Café': 'str', 'Cocktail Bar': 'str', 'Coffee Shop': 'str', 'Deli / Bodega': 'str', 'Diner': 'str', 'Dive Bar': 'str', 'Eastern European Restaurant': 'str', 'Fast Food Restaurant': 'str', 'Food & Drink Shop': 'str', 'Food Truck': 'str', 'French Restaurant': 'str', 'Fried Chicken Joint': 'str', 'Gastropub': 'str', 'Greek Restaurant': 'str', 'Hookah Bar': 'str', 'Hot Dog Joint': 'str', 'Hotel Bar': 'str', 'Irish Pub': 'str', 'Italian Restaurant': 'str', 'Juice Bar': 'str', 'Lounge': 'str', 'Mediterranean Restaurant': 'str', 'New American Restaurant': 'str', 'Piano Bar': 'str', 'Pizza Place': 'str', 'Pub': 'str', 'Restaurant': 'str', 'Rock Club': 'str', 'Sandwich Place': 'str', 'Seafood Restaurant': 'str', 'Southern / Soul Food Restaurant': 'str', 'Sports Bar': 'str', 'Steakhouse': 'str', 'Theme Restaurant': 'str', 'Vegetarian / Vegan Restaurant': 'str', 'Whisky Bar': 'str', 'Wings Joint': 'str', 'Bath Beach': 'str', 'Bay Ridge': 'str', 'Baychester': 'str', 'Bedford Park': 'str', 'Bedford Stuyvesant': 'str', 'Belmont': 'str', 'Boerum Hill': 'str', 'Brighton Beach': 'str', 'Brooklyn Heights': 'str', 'Bushwick': 'str', 'Carroll Gardens': 'str', 'Central Harlem': 'str', 'Chinatown': 'str',
              'City Island': 'str', 'City Line': 'str', 'Clinton': 'str', 'Clinton Hill': 'str', 'Co-op City': 'str', 'Cobble Hill': 'str', 'Crown Heights': 'str', 'Downtown': 'str', 'Dyker Heights': 'str', 'East Harlem': 'str', 'East Tremont': 'str', 'East Williamsburg': 'str', 'Eastchester': 'str', 'Edgewater Park': 'str', 'Flatlands': 'str', 'Fordham': 'str', 'Fort Greene': 'str', 'Fort Hamilton': 'str', 'Georgetown': 'str', 'Gerritsen Beach': 'str', 'Gowanus': 'str', 'Greenpoint': 'str', 'Hamilton Heights': 'str', 'High  Bridge': 'str', 'Inwood': 'str', 'Kensington': 'str', 'Kingsbridge': 'str', 'Lenox Hill': 'str', 'Lincoln Square': 'str', 'Manhattan Beach': 'str', 'Manhattanville': 'str', 'Marble Hill': 'str', 'Melrose': 'str', 'Midtown': 'str', 'Morris Park': 'str', 'Morrisania': 'str', 'Mott Haven': 'str', 'Navy Yard': 'str', 'North Riverdale': 'str', 'North Side': 'str', 'Norwood': 'str', 'Park Slope': 'str', 'Parkchester': 'str', 'Prospect Heights': 'str', 'Prospect Lefferts Gardens': 'str', 'Prospect Park South': 'str', 'Red Hook': 'str', 'Schuylerville': 'str', 'Sheepshead Bay': 'str', 'Soundview': 'str', 'South Side': 'str', 'Spring Creek': 'str', 'Throgs Neck': 'str', 'Upper East Side': 'str', 'Upper West Side': 'str', 'Van Nest': 'str', 'Washington Heights': 'str', 'Westchester Square': 'str', 'Williamsburg': 'str', 'Windsor Terrace': 'str', 'Woodlawn': 'str', 'Yorkville': 'str', 'Tip Count': 'float64', 'Rating': 'float64', 'Like Count': 'str'}

    df = pd.read_csv(args.data, parse_dates=True,
                     usecols=cols,
                     dtype=dtypes)

    if args.mode == 'train':
        build_encoders(df)
        encoders = load_encoders()
        model_train(df, encoders, args)
    elif args.mode == 'predict':
        encoders = load_encoders()
        model = xgb.Booster()
        model.load_model('model.bin')
        predictions = model_predict(df, model, encoders)
        if args.type == 'csv':
            predictions.to_csv('predictions.csv', index=False)
        if args.type == 'json':
            with open('predictions.json', 'w', encoding='utf-8') as f:
                f.write(predictions.to_json(orient="records"))
