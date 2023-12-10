from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion


import xgboost as xgb
import lightgbm as lgb


# Read Data

train = pd.read_parquet(Path("/kaggle/input/mdsb-2023/train.parquet"))
test = pd.read_parquet(Path("/kaggle/input/mdsb-2023/final_test.parquet"))

# Data Preprocessing:
class DateEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["year"] = X["date"].dt.year
        X["month"] = X["date"].dt.month
        X["day"] = X["date"].dt.day
        X["weekday"] = X["date"].dt.weekday
        X["hour"] = X["date"].dt.hour
        return X


# Categort variables encoder to integers(for ml)
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.label_encoder = LabelEncoder()
        # Fit the LabelEncoder on the first column assuming it's the target column for encoding
        self.label_encoder.fit(X.iloc[:, 0])
        return self

    def transform(self, X):
        X_encoded = pd.DataFrame(self.label_encoder.transform(X.iloc[:, 0]), columns=[X.columns[0]])
        return X_encoded


def create_is_weekday(X):
    # Assuming 'weekday' is the 5th column after preprocessing
    is_weekday = (X[:, 4] < 5).astype(int)
    return is_weekday.reshape(-1, 1)


preprocessor_1 = ColumnTransformer(
    transformers=[
        ("date_encoder", DateEncoder(), ["date"]),
        ("label_encoder", LabelEncoderTransformer(), ["counter_id"]),
        ('drop_columns', 'drop',
         ['counter_technical_id', 'site_id', 'bike_count', 'coordinates', 'counter_name', 'counter_installation_date'])
    ],
    remainder='passthrough'  # Keep columns not specified in transformers
)


def create_season(X):
    month_index = 2  # Replace with the correct index of the 'month' column
    months = X[:, month_index].astype(int)

    # Determine the season based on the month
    seasons = np.array([
        '1' if 3 <= month <= 5 else
        '2' if 6 <= month <= 8 else
        '3' if 9 <= month <= 11 else
        '4'
        for month in months
    ])
    return seasons.reshape(-1, 1)


pipeline_1 = Pipeline(
    steps=[
        ("preprocessor_1", preprocessor_1),
        ("feature_union", FeatureUnion([
            ("passthrough", FunctionTransformer(lambda x: x)),
            ("is_weekday", FunctionTransformer(create_is_weekday, validate=False)),
            ("season", FunctionTransformer(create_season, validate=False))
        ]))
    ]
)

preprocessor_2 = ColumnTransformer(
    transformers=[
        ("date_encoder", DateEncoder(), ["date"]),
        ("label_encoder", LabelEncoderTransformer(), ["counter_id"]),
        ('drop_columns', 'drop',
         ['counter_technical_id', 'site_id', 'counter_name', 'counter_installation_date', 'coordinates'])
    ],
    remainder='passthrough'  # Keep columns not specified in transformers
)

pipeline_2 = Pipeline(
    steps=[
        ("preprocessor_2", preprocessor_2),
        ("feature_union", FeatureUnion([
            ("passthrough", FunctionTransformer(lambda x: x)),
            ("is_weekday", FunctionTransformer(create_is_weekday, validate=False)),
            ("season", FunctionTransformer(create_season, validate=False))
        ]))
    ]
)

preprocessor_3 = ColumnTransformer(
    transformers=[
        ("date_encoder", DateEncoder(), ["date"]),
        ("label_encoder", LabelEncoderTransformer(), ["counter_id"]),
        ('drop_columns', 'drop',
         ['counter_technical_id', 'site_id', 'counter_name', 'counter_installation_date', 'bike_count'])
    ],
    remainder='passthrough'  # Keep columns not specified in transformers
)

pipeline_3 = Pipeline(
    steps=[
        ("preprocessor_3", preprocessor_3),
        ("feature_union", FeatureUnion([
            ("passthrough", FunctionTransformer(lambda x: x)),
            ("is_weekday", FunctionTransformer(create_is_weekday, validate=False)),
            ("season", FunctionTransformer(create_season, validate=False))
        ]))
    ]
)

transformed_data = pipeline_1.fit_transform(train)

columns = ["date", "year", "month", "day", "weekday", "hour", "counter_id", "site_name", "latitude", "longitude",
           "log_bike_count", "is_weekday", "season"]

train_data = pd.DataFrame(transformed_data, columns=columns)

train_data['log_bike_count'] = train_data['log_bike_count'].astype(float)



#tran = pd.read_csv('/kaggle/input/testdata/transportation_lines.csv')
tran = pd.read_csv(Path("/kaggle/input/externals/transportation_lines.csv"))
train = pd.merge(train_data, tran, on='site_name', how='left')
train.dropna(subset=['log_bike_count'], inplace=True)

train_data = train
train_data.loc[train_data['counter_id'].isin([40, 41]), 'Lines'] = 5

# Specific arr values for each counter_id
arr_mapping = {
    16: [34, 35, 6, 7, 32, 33],
    15: [54, 55, 15, 16],
    14: [19, 20],
    7: [23, 25, 40, 41, 50, 51],
    8: [24, 46, 47],
    12: [26, 27, 13, 14, 2, 3, 37, 36, 0, 1],
    13: [4, 30, 31, 44, 45],
    5: [4, 5, 42, 43],
    2: [22, 21, 49, 48, 39, 38],
    11: [8],
    20: [29, 28],
    17: [53, 52],
    19: [12, 11, 10, 9, 18, 17]
}

train_data['arr'] = None

for arr, counter_ids in arr_mapping.items():
    train_data.loc[train_data['counter_id'].isin(counter_ids), 'arr'] = arr

# The influence of Covid-19

# In[81]:


# Covid-19
mask1 = (train_data['date'] >= '2020-03-17 12:00:00') & (train_data['date'] <= '2020-06-15')
mask2 = (train_data['date'] >= '2020-10-30') & (train_data['date'] <= '2020-12-15')
mask3 = (train_data['date'] >= '2021-03-19') & (train_data['date'] <= '2021-05-18')
train_data['is_confinement'] = 0
train_data.loc[mask1 | mask2 | mask3, 'is_confinement'] = 1



class DropAllNaColumns(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna(axis=1, how='all')


class DropSingleValueColumns(TransformerMixin):
    def fit(self, X, y=None):
        self.columns_to_keep = X.columns[X.nunique(dropna=True) > 1]
        return self

    def transform(self, X):
        return X[self.columns_to_keep]


class DropDuplicateDates(TransformerMixin):
    def fit(self, X, y=None):
        self.unique_dates = X['date'].drop_duplicates()
        return self

    def transform(self, X):
        return X[X['date'].isin(self.unique_dates)]


# Create the pipeline
external_data_pipeline = Pipeline([
    ('drop_all_na_columns', DropAllNaColumns()),
    ('drop_single_value_columns', DropSingleValueColumns()),
    ('drop_duplicate_dates', DropDuplicateDates())
])

# Weather influences

weather_data = pd.read_csv('/kaggle/input/externals/weather.csv')
weather = external_data_pipeline.fit_transform(weather_data)
weather['date'] = pd.to_datetime(weather['date'])

columns_to_cap = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'cloud_cover (%)', 'wind_speed_10m (km/h)']

for column in columns_to_cap:
    if column != 'date' and weather[column].dtype != 'object':  # Skip non-numeric columns and 'date' column
        lower_threshold = weather[column].quantile(0.01)
        upper_threshold = weather[column].quantile(0.99)

        weather[column] = np.where(weather[column] < lower_threshold, lower_threshold, weather[column])
        weather[column] = np.where(weather[column] > upper_threshold, upper_threshold, weather[column])

columns_to_scale = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'cloud_cover (%)', 'wind_speed_10m (km/h)']

scaler = StandardScaler()

weather[columns_to_scale] = scaler.fit_transform(weather[columns_to_scale])
weather.info()


# Merge
merged_train = pd.merge(train_data, weather, on='date', how='left')
merged_train.dropna(subset=['log_bike_count'], inplace=True)


final_test_data = pipeline_2.fit_transform(test)
columns = ['date', "year", "month", "day", "weekday", "hour", "counter_id", "site_name", "latitude", "longitude",
           "is_weekday", 'season']
final_test = pd.DataFrame(final_test_data, columns=columns)

final_test = pd.merge(final_test, tran, on='site_name', how='left')
final_test.loc[final_test['counter_id'].isin([40, 41]), 'Lines'] = 5
transformed_test = final_test.drop(columns=['site_name'])

transformed_test['arr'] = None

# Update 'arr' column based on arr_mapping
for arr, counter_ids in arr_mapping.items():
    transformed_test.loc[transformed_test['counter_id'].isin(counter_ids), 'arr'] = arr

test_data = transformed_test

test_data['is_confinement'] = 0

merged_test = pd.merge(test_data, weather, on='date', how='left')

merged_test['original_index'] = merged_test.index
merged_test = merged_test.sort_values(by=["date"]).reset_index(drop=True)




# # Modeling


# Categorize the Various Features
cat_feats = ["year", "hour", "counter_id", "latitude", "longitude", "weekday", "is_weekday", "is_confinement", "is_day",
             "month", "season", "arr"]
num_feats = ["Lines", "temperature_2m (°C)", "relative_humidity_2m (%)", "cloud_cover (%)", "wind_speed_10m (km/h)"]

# Force Convert Categorical Features to Category
for var in cat_feats:
    merged_train[var] = merged_train[var].astype("category")

data_train = merged_train.sort_values(by=["date"]).reset_index(drop=True)

# Specify Features of the Training Set, Corresponding Labels, and Features of the Test Set
X_train = data_train[cat_feats + num_feats]
y_train = data_train["log_bike_count"].values
X_test = merged_test[cat_feats + num_feats]
test_id = merged_test["date"]


scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split the Dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

# RMSE in the original scale
from sklearn.metrics import mean_squared_error
import numpy as np


def rmse(y_true_log, y_pred_log):
    rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    return rmse


# In[ ]:


# In[40]:


# Define a Parameter Tuning Function
def find_best_params(model, params, cv=5, n_jobs=-1, X_train=X_train, label=False):
    """
    Use Grid Search with Cross-Validation (GridSearchCV) for Parameter Tuning, Return the Best Model


    """
    rmse_scorer = metrics.make_scorer(rmse, greater_is_better=False)  # Define the Scoring Function for GridSearchCV
    grid_cv = GridSearchCV(model, param_grid=params, scoring=rmse_scorer, n_jobs=n_jobs, cv=cv, verbose=2)
    grid_cv.fit(X_train, y_train)
    if label:
        # Visualize the RMSE Scores for Each Parameter Tuning
        fig, ax = plt.subplots(figsize=(12, 5))
        df = pd.DataFrame(grid_cv.cv_results_)
        df["alpha"] = df["params"].apply(lambda x: round(x["alpha"], 3))
        df["rmse"] = df["mean_test_score"].apply(lambda x: -round(x, 4))
        sns.pointplot(data=df, x="alpha", y="rmse", ax=ax)
    # Output the Best RMSLE Score and the Optimal Parameters
    print("The best RMSE score is: %.3f" % (-grid_cv.best_score_))
    print("The best parameter is: %s" % (grid_cv.best_params_))
    return grid_cv.best_estimator_

# Define RMSE Result Data Frame
global i
i = 0

def result_save(y_val, y_pred, label):
    result_df = pd.DataFrame({"Model": label, "RMSE": rmse(y_val, y_pred)}, index=[i])
    return result_df


# Apply Model for Fitting and Prediction.
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_val)
print("RMSE For LinearRegression: %.3f" % (rmse(y_val, y_pred)))

# Save RMSE
labels = []
label = "Linear Regression"
labels.append(label)
i = 0
result = result_save(y_val, y_pred, label)
result_df = result
del result


# #### Logistic Regression
import pandas as pd

# Apply Model for Fitting and Prediction.
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train.astype('int'))
y_pred = lr_model.predict(X_val)
print("RMSE For Logistic Regression: %.3f" % (rmse(y_val, y_pred)))

label = "Logistic Regression"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result


# #### Lasso Regression

get_ipython().run_cell_magic('time', '',
                             "#Use GridSearchCV for Parameter Tuning\nparams = {\n    'alpha': 1 / np.array([0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000])\n}\nlasso_model = find_best_params(Lasso(), params, label=True)\nlasso_model\n")

# Use the Optimally Tuned Model for Fitting and Prediction
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_val)
print("RMSE For Lasso Regression: %.3f" % (rmse(y_val, y_pred)))

label = "Lasso Regression"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result

# #### Ridge Regression

get_ipython().run_cell_magic('time', '',
                             "\nparams = {\n    'alpha': [0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000]\n}\nridge_model = find_best_params(Ridge(), params, label=True)\nridge_model\n")

ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_val)
print("RMSE For Ridge Regression: %.3f" % (rmse(y_val, y_pred)))

label = "Ridge Regression"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result


# #### ElasticNet

get_ipython().run_cell_magic('time', '',
                             "\nparams = {\n    'alpha': [0.1, 0.01, 0.005, 0.0025, 0.001],\n    'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.8]\n}\nenet_model = find_best_params(ElasticNet(), params, label=False)\nenet_model\n")

enet_model.fit(X_train, y_train)
y_pred = enet_model.predict(X_val)
print("RMSE For ElasticNet Regression: %.3f" % (rmse(y_val, y_pred)))

label = "ElasticNet Regression"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result


# #### DecisionTreeRegressor

dtr_model = DecisionTreeRegressor()
dtr_model.fit(X_train, y_train)
y_pred = dtr_model.predict(X_val)
print("RMSE For DecisionTreeRegressor: %.3f" % (rmse(y_val, y_pred)))

label = "DecisionTreeRegressor"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result


# #### ExtraTreeRegressor

etr_model = ExtraTreeRegressor()
etr_model.fit(X_train, y_train)
y_pred = etr_model.predict(X_val)
print("RMSE For ExtraTreeRegressor: %.3f" % (rmse(y_val, y_pred)))

label = "ExtraTreeRegressor"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result


# #### BaggingRegressor

bagging_model = BaggingRegressor()
bagging_model.fit(X_train, y_train)
y_pred = bagging_model.predict(X_val)
print("RMSE For BaggingRegressor: %.3f" % (rmse(y_val, y_pred)))

label = "BaggingRegressor"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result


# #### RandomForestRegressor

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_val)
print("RMSE For RandomForestRegressor: %.3f" % (rmse(y_val, y_pred)))

label = "RandomForestRegressor"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result


# #### AdaBoostRegressor

ada_model = AdaBoostRegressor()
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_val)
print("RMSE For AdaBoostRegressor: %.3f" % (rmse(y_val, y_pred)))

label = "AdaBoostRegressor"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result


# #### GBRT

gbrt_model = GradientBoostingRegressor()
gbrt_model.fit(X_train, y_train)
y_pred = gbrt_model.predict(X_val)
print("RMSE For GBRT: %.3f" % (rmse(y_val, y_pred)))

label = "GBRT"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result


# #### XGBoost

xgb_model = xgb.XGBRFRegressor(n_estimators=1000, max_depth=9)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_val)
print("RMSE For XGBoost: %.3f" % (rmse(y_val, y_pred)))

label = "XGBoost"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result


# #### LightGBM

lgb_model = lgb.LGBMRegressor(n_estimators=1000)
lgb_model.fit(X_train, y_train)
y_pred = lgb_model.predict(X_val)
print("RMSE For LightGBM: %.3f" % (rmse(y_val, y_pred)))

label = "LightGBM"
labels.append(label)
i += 1
result = result_save(y_val, y_pred, label)
result_df = result_df._append(result)
del result

result_df.sort_values(by='RMSE', ascending=True)

estimators = [('lgb', lgb.LGBMRegressor(n_estimators=1000)),
              ('rf', RandomForestRegressor()),
              ('bagging', BaggingRegressor()),
              ('dst', DecisionTreeRegressor())
              ]

stacking_model = StackingRegressor(estimators=estimators,
                                   final_estimator=GradientBoostingRegressor(random_state=42))
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_val)
print("RMSE For Stacking: %.4f" % (rmse(y_val, y_pred)))

# # Predictition



stacking_pred = stacking_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
ensemble = stacking_pred * 0.60 + lgb_pred * 0.40

predictions_df = pd.DataFrame({
    'original_index': merged_test['original_index'],
    'log_bike_count': ensemble
})
submission_df = predictions_df.sort_values(by='original_index').reset_index(drop=True)
# Create a final submission DataFrame with 'Id' and 'log_bike_count'
final_submission = submission_df.rename(columns={'original_index': 'Id'})

class PredictionTransformer:
    def transform(self, predictions):
        # Initialize variables
        a1 = np.log(1)
        upper_bound = 0.025
        predictions = np.where(predictions < upper_bound, a1, predictions)

        bounds = [
            (np.log(2), 0.025, 0.696),
            (np.log(3), 0.696, 1.1),
            (np.log(4), 1.1, 1.388),
            (np.log(5), 1.388, 1.6096),
            (np.log(6), 1.6096, 1.793),
            (np.log(7), 1.793, 1.946),
        ]
        for a, lower_bound, upper_bound in bounds:
            predictions = np.where((predictions > lower_bound) & (predictions <= upper_bound), a, predictions)

        lower_bound = 1.946

        for i in range(8, 30):
            log_i = np.log(i)
            log_i_plus_1 = np.log(i + 1)
            upper_bound = log_i + 0.001 * (log_i_plus_1 - log_i)
            predictions = np.where((predictions > lower_bound) & (predictions <= upper_bound), log_i, predictions)
            lower_bound = upper_bound

        return predictions

transformer = PredictionTransformer()

predictions = transformer.transform(final_submission)
# Save the DataFrame to a CSV file
predictions.to_csv('submission.csv', index=False)

