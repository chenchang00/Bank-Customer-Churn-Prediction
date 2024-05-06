from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# One-Hot: Geography
# Binary: Gender
# Min-Max: Age, Tenure, NumofProducts, Satisfaction Score, Balance, EstimatedSalary, PointEarned, CreditScore
# Transform: Card_Type
# Do Nothing: HasCrCard, IsActiveMember, Complain, 
class type_to_num(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.replace("DIAMOND",1).replace("PLATINUM",0.75).replace("GOLD", 0.5).replace("SILVER",0.25)

class DoNothing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
 
def data_transformation(data):
    continuous_cols = ['Age','Tenure','NumOfProducts','Satisfaction Score','Balance','EstimatedSalary','Point Earned','CreditScore']
    cat_cols = ['Geography']
    binary_cols = ['Gender']
    card_type = ['Card Type']
    remain_unchanged = ['HasCrCard','IsActiveMember','Complain']

    continuous_transformer = Pipeline(steps=[
        ('MinMax', MinMaxScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    binary_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='if_binary'))
    ])
    card_tranformer = Pipeline(steps=[
        ('extract_first_digit', type_to_num())
    ])
    Nothing = Pipeline(steps=[
        ('do nothing', DoNothing())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_transformer, continuous_cols),
            ('cat', cat_transformer, cat_cols),
            ('bin', binary_transformer, binary_cols),
            ('card', card_tranformer, card_type),
            ('Nothing', Nothing, remain_unchanged)
        ]
    )
    pipe = Pipeline(steps=[('pre',preprocessor)])
    pipe.fit(data)
    transformed_data = pipe.transform(data)
    return transformed_data