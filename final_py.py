from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, StackingRegressor, GradientBoostingRegressor

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
import lightgbm as lgb


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


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(X.iloc[:, 0])
        return self

    def transform(self, X):
        X_encoded = pd.DataFrame(self.label_encoder.transform(X.iloc[:, 0]), columns=[X.columns[0]])
        return X_encoded


def create_is_weekday(X):
    # Assuming 'weekday' is the 5th column after preprocessing
    is_weekday = (X[:, 4] < 5).astype(int)
    return is_weekday.reshape(-1, 1)


def create_season(X):
    month_index = 2
    months = X[:, month_index].astype(int)

    # Determine the season based on the month
    seasons = np.array([
        '2' if 3 <= month <= 5 else
        '3' if 6 <= month <= 8 else
        '4' if 9 <= month <= 11 else
        '1'
        for month in months
    ])
    return seasons.reshape(-1, 1)


def convert_log_bike_count(X):
    X['log_bike_count'] = X['log_bike_count'].astype(float)
    return X


class ToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)


def create_season(X):
    month_index = 2
    months = X[:, month_index].astype(int)
    seasons = np.array([
        '1' if 3 <= month <= 5 else
        '2' if 6 <= month <= 8 else
        '3' if 9 <= month <= 11 else
        '4'
        for month in months
    ])
    return seasons.reshape(-1, 1)


class HourlyMeanRankTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        hourly_mean = train_data.groupby('hour')['log_bike_count'].mean().reset_index()
        hourly_mean['rank'] = hourly_mean['log_bike_count'].rank(ascending=False, method='dense')
        return X.merge(hourly_mean[['hour', 'rank']], on='hour', how='left')


class MonthlyMeanRankTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        monthly_mean = train_data.groupby('month')['log_bike_count'].mean().reset_index()
        monthly_mean['rank_month'] = monthly_mean['log_bike_count'].rank(ascending=False, method='dense')
        return X.merge(monthly_mean[['month', 'rank_month']], on='month', how='left')


class MergeDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dataframe, on, how='left'):
        self.dataframe = dataframe
        self.on = on
        self.how = how

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.merge(X, self.dataframe, on=self.on, how=self.how)


class ArrayMappingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['arr'] = None
        for arr, counter_ids in self.mapping.items():
            X.loc[X['counter_id'].isin(counter_ids), 'arr'] = arr
        X = X.drop(columns=['site_name'])
        return X


class ConfinementFlagTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mask1 = (X['date'] >= '2020-03-17 12:00:00') & (X['date'] <= '2020-06-15')
        mask2 = (X['date'] >= '2020-10-30') & (X['date'] <= '2020-12-15')
        mask3 = (X['date'] >= '2021-03-19') & (X['date'] <= '2021-05-18')
        X['is_confinement'] = 0
        X.loc[mask1 | mask2 | mask3, 'is_confinement'] = 1
        return X


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


class DropDuplicatesTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop_duplicates()


class WeatherDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_cap, columns_to_scale):
        self.columns_to_cap = columns_to_cap
        self.columns_to_scale = columns_to_scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['date'] = pd.to_datetime(X['date'])

        for column in self.columns_to_cap:
            if column != 'date' and X[column].dtype != 'object':
                lower_threshold = X[column].quantile(0.01)
                upper_threshold = X[column].quantile(0.99)

                X[column] = np.where(X[column] < lower_threshold, lower_threshold, X[column])
                X[column] = np.where(X[column] > upper_threshold, upper_threshold, X[column])

        scaler = StandardScaler()
        X[self.columns_to_scale] = scaler.fit_transform(X[self.columns_to_scale])

        X['is_day'] = X['is_day'].astype(str)

        return X


class MergeWeatherTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, weather_data):
        self.weather_data = weather_data

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = pd.merge(X, self.weather_data, on='date', how='left')
        return X


class CustomPredictionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, predictions):
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
            predictions = np.where((predictions > lower_bound) & (predictions < upper_bound), a, predictions)

        for i in range(8, 30):
            log_i = np.log(i)
            log_i_plus_1 = np.log(i + 1)
            upper_bound = log_i + 0.001 * (log_i_plus_1 - log_i)
            predictions = np.where((predictions > lower_bound) & (predictions < upper_bound), log_i, predictions)
            lower_bound = upper_bound

        return predictions


def create_categorical_transformer():
    return Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


def create_model_pipeline(categorical_features, numeric_features, X_train, y_train):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', create_categorical_transformer(), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])

    xgb_model = XGBRegressor(n_estimators=350, learning_rate=0.1, max_depth=12, random_state=42)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline


preprocessor_train = ColumnTransformer(
    transformers=[
        ("date_encoder", DateEncoder(), ["date"]),
        ("label_encoder", LabelEncoderTransformer(), ["counter_id"]),
        ('drop_columns', 'drop', ['counter_technical_id', 'site_id', 'bike_count', 'coordinates', 'latitude', 'longitude', 'counter_name', 'counter_installation_date'])
    ],
    remainder='passthrough'
)


pipeline_train = Pipeline(
    steps=[
        ("convert_log_bike_count", FunctionTransformer(convert_log_bike_count, validate=False)),
        ("preprocessor_train", preprocessor_train),
        ("feature_union", FeatureUnion([
            ("passthrough", FunctionTransformer(lambda x: x)),
            ("is_weekday", FunctionTransformer(create_is_weekday, validate=False)),
            ("season", FunctionTransformer(create_season, validate=False))
        ])),
        ("to_dataframe", ToDataFrame(columns=['date',"year", "month", "day", "weekday", "hour", "counter_id", "site_name", "log_bike_count", "is_weekday", "season"]))
    ]
)

preprocessor_test = ColumnTransformer(
    transformers=[
        ("date_encoder", DateEncoder(), ["date"]),
        ("label_encoder", LabelEncoderTransformer(), ["counter_id"]),
        ('drop_columns', 'drop', ['counter_technical_id', 'site_id', 'counter_name', 'counter_installation_date','coordinates','latitude','longitude'])
    ],
    remainder='passthrough'
)


pipeline_test = Pipeline(
    steps=[
        ("preprocessor_test", preprocessor_test),
        ("feature_union", FeatureUnion([
            ("passthrough", FunctionTransformer(lambda x: x)),
            ("is_weekday", FunctionTransformer(create_is_weekday, validate=False)),
            ("season", FunctionTransformer(create_season, validate=False))
        ])),
        ("to_dataframe", ToDataFrame(columns=['date',"year", "month", "day", "weekday", "hour", "counter_id", "site_name", "is_weekday",'season']))
    ]
)


external_pre_pipeline = Pipeline([
    ('drop_all_na_columns', DropAllNaColumns()),
    ('drop_single_value_columns', DropSingleValueColumns()),
    ('drop_duplicates', DropDuplicatesTransformer()),
])


columns_to_cap = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'cloud_cover (%)', 'wind_speed_10m (km/h)']
columns_to_scale = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'cloud_cover (%)', 'wind_speed_10m (km/h)']


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


#tran = pd.read_csv('data/transportation_lines.csv')
tran = pd.read_csv(Path("/kaggle/input/externals/transportation_lines.csv"))
tran = external_pre_pipeline.fit_transform(tran)

#weather_data = pd.read_csv('data/weather.csv')
weather_data = pd.read_csv('/kaggle/input/externals/weather.csv')
weather_transformer = WeatherDataTransformer(columns_to_cap, columns_to_scale)
weather = weather_transformer.fit_transform(weather_data)
weather = external_pre_pipeline.fit_transform(weather)


pipeline_external = Pipeline([
    ('hourly_mean_rank', HourlyMeanRankTransformer()),
    ('monthly_mean_rank', MonthlyMeanRankTransformer()),
    ('merge_data', MergeDataTransformer(tran, on='site_name')),
    ('array_mapping', ArrayMappingTransformer(arr_mapping)),
    ('confinement_flag', ConfinementFlagTransformer()),
    ('merge_weather', MergeWeatherTransformer(weather))
])

#data = pd.read_parquet(Path("data/train.parquet"))
data = pd.read_parquet(Path("/kaggle/input/mdsb-2023/train.parquet"))

train_data = pipeline_train.fit_transform(data)
train_data.sort_values('date', inplace=True)

#final_test = pd.read_parquet(Path("data/final_test.parquet"))
final_test = pd.read_parquet(Path("/kaggle/input/mdsb-2023/final_test.parquet"))

test_data = pipeline_test.fit_transform(final_test)

train_data = pipeline_external.fit_transform(train_data)
test_data = pipeline_external.fit_transform(test_data)

test_data['original_index'] = test_data.index
test_data = test_data.sort_values(by=["date"]).reset_index(drop=True)


# Categorize the Various Features
cat_feats = ["year", "hour", "counter_id", "weekday", "is_weekday", "is_confinement", "is_day",
             "month", "season", "arr"]
num_feats = ["Lines", "temperature_2m (°C)", "relative_humidity_2m (%)", "cloud_cover (%)", "wind_speed_10m (km/h)"]

# Force Convert Categorical Features to Category
for var in cat_feats:
    train_data[var] = train_data[var].astype("category")

data_train = train_data.sort_values(by=["date"]).reset_index(drop=True)

# Specify Features of the Training Set, Corresponding Labels, and Features of the Test Set
X_train = data_train[cat_feats + num_feats]
y_train = data_train["log_bike_count"].values
X_test = test_data[cat_feats + num_feats]
test_id = test_data["date"]


scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# #### DecisionTreeRegressor

dtr_model = DecisionTreeRegressor()
dtr_model.fit(X_train, y_train)

# #### BaggingRegressor

bagging_model = BaggingRegressor()
bagging_model.fit(X_train, y_train)


# #### RandomForestRegressor

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# #### LightGBM

lgb_model = lgb.LGBMRegressor(n_estimators=1000)
lgb_model.fit(X_train, y_train)

estimators = [('lgb', lgb.LGBMRegressor(n_estimators=1000)),
              ('rf', RandomForestRegressor()),
              ('bagging', BaggingRegressor()),
              ('dst', DecisionTreeRegressor())
              ]

stacking_model = StackingRegressor(estimators=estimators,
                                   final_estimator=GradientBoostingRegressor(random_state=42))
stacking_model.fit(X_train, y_train)

# # Predictition

stacking_pred = stacking_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)
ensemble = stacking_pred * 0.60 + lgb_pred * 0.40

predictions_df = pd.DataFrame({
    'original_index': test_data['original_index'],
    'log_bike_count': ensemble
})
submission_df = predictions_df.sort_values(by='original_index').reset_index(drop=True)
# Create a final submission DataFrame with 'Id' and 'log_bike_count'
final_submission = submission_df.rename(columns={'original_index': 'Id'})
final_submission.to_csv('submission.csv', index=False)
