import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import sqlite3

def load_makes_from_db(db_name="cars.db"):
    conn = sqlite3.connect(db_name)
    query = "SELECT DISTINCT make FROM cars"
    makes = pd.read_sql(query, conn)
    conn.close()
    return makes

def load_models_from_db(make, db_name="cars.db", min_entries=20):
    conn = sqlite3.connect(db_name)
    query = f"""
    SELECT model, COUNT(*) as count
    FROM cars
    WHERE make = '{make}'
    GROUP BY model
    HAVING COUNT(*) > {min_entries}
    """
    models = pd.read_sql(query, conn)
    conn.close()
    return models

def load_data_from_db(make, model=None, db_name="cars.db"):
    conn = sqlite3.connect(db_name)
    if model and model != "0":
        query = f"SELECT first_registration, make, mileage, model, price, fuel_type, link, color FROM cars WHERE make = '{make}' AND model = '{model}'"
    else:
        query = f"SELECT first_registration, make, mileage, model, price, fuel_type, link, color FROM cars WHERE make = '{make}'"
    car_data = pd.read_sql(query, conn)
    conn.close()
    return car_data


# Step 1: Display all unique makes and prompt user to select one
makes = load_makes_from_db()
print("Select a car brand by typing the corresponding number:")
for idx, make in enumerate(makes['make'], start=1):
    print(f"{idx}. {make}")

make_choice = int(input("Enter the number corresponding to the brand: ")) - 1
selected_make = makes['make'].iloc[make_choice]

# Step 2: Display all models for the selected make and prompt user to select a type (or all)
models = load_models_from_db(selected_make)
print(f"\nSelect a model for {selected_make} by typing the corresponding number, or type 0 for all models:")
for idx, model in enumerate(models['model'], start=1):
    print(f"{idx}. {model}")

model_choice = input("Enter the number corresponding to the model, or type 0 for all models: ")
if model_choice == "0":
    selected_model = None
else:
    selected_model = models['model'].iloc[int(model_choice) - 1]

# Step 3: Load data from the database based on user input
car_data = load_data_from_db(selected_make, selected_model)
car_data['first_registration'] = car_data['first_registration'].replace('new', 2024)
car_data['first_registration'] = pd.to_numeric(car_data['first_registration'], errors='coerce')
car_data['car_age'] = 2024 - car_data['first_registration']

X = car_data[['car_age', 'make', 'mileage', 'model', 'fuel_type', 'link', 'color']]
y = car_data['price']

encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X[['make', 'model', 'fuel_type']])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['make', 'model', 'fuel_type']))
X_encoded_full = pd.concat([X_encoded_df, X[['car_age', 'mileage']].reset_index(drop=True)], axis=1)

links = X['link'].reset_index(drop=True)
car_age = X['car_age'].reset_index(drop=True)
makes = X['make'].reset_index(drop=True)
models = X['model'].reset_index(drop=True)
mileage = X['mileage'].reset_index(drop=True)
color = X['color'].reset_index(drop=True)

X_encoded_full.columns = X_encoded_full.columns.astype(str)

kf = KFold(n_splits=5)
rf_model = RandomForestRegressor(random_state=42)
all_results = []
mae_list = []
mape_list = []

for train_index, test_index in kf.split(X_encoded_full):
    X_train, X_test = X_encoded_full.iloc[train_index], X_encoded_full.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    links_test = links.iloc[test_index].reset_index(drop=True)
    color_test = color.iloc[test_index].reset_index(drop=True)
    car_age_test = car_age.iloc[test_index].reset_index(drop=True)
    makes_test = makes.iloc[test_index].reset_index(drop=True)
    models_test = models.iloc[test_index].reset_index(drop=True)
    mileage_test = mileage.iloc[test_index].reset_index(drop=True)

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test).round().astype(int)

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    mae_list.append(mae)
    mape_list.append(mape)

    print(f"Fold Results:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

    results_df = pd.DataFrame({
        'Actual Price': y_test.reset_index(drop=True),
        'Predicted Price': y_pred,
        'Year': 2024 - car_age_test,
        'Make': makes_test,
        'Model': models_test,
        'Mileage': mileage_test,
        'color': color_test,
        'Link': links_test
    })

    results_df['Actual Price'] = pd.to_numeric(results_df['Actual Price'], errors='coerce')
    results_df['Predicted Price'] = pd.to_numeric(results_df['Predicted Price'], errors='coerce')

    filtered_results = results_df[results_df['Actual Price'] <= results_df['Predicted Price'] * 0.75]
    all_results.append(filtered_results)

final_results = pd.concat(all_results, ignore_index=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

final_results['Actual/Predicted'] = final_results['Actual Price'] / final_results['Predicted Price']
final_results_sorted = final_results.sort_values(by='Actual/Predicted', ascending=True)
print(final_results_sorted)

combined_mae = sum(mae_list) / len(mae_list)
combined_mape = sum(mape_list) / len(mape_list)

print(f"\nCombined MAE (across all folds): {combined_mae}")
print(f"Combined MAPE (across all folds): {combined_mape}")

def create_db():
    sqlite3.connect('cars.db')
