##### Libraries #####
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from sklearn import set_config
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, PolynomialFeatures,
                                   PowerTransformer, StandardScaler)
from skopt import BayesSearchCV

##### Import Data #####
df = pd.read_csv('datasets/Data_Science_Challenge.csv')

# data types of columns
def df_datatypes(df):
    df_desc = pd.DataFrame(df.dtypes.value_counts().reset_index())
    df_desc.columns = ['Data Type', 'Count']
    return df_desc.sort_values('Count', ascending=False)

df_datatypes(df)

# categorical - describe
def df_describe_categorical(df):
    return df.select_dtypes(exclude=np.number).describe()

df_describe_categorical(df)

# numerical - describe
def df_describe_numerical(df):
    return df.select_dtypes(include=np.number).describe()

df_describe_numerical(df)

# numerical - correlogram
def graph_correlogram(df):
    sns.set_theme(style="white") 
    # Compute the correlation matrix
    corr = df.corr(numeric_only=True)
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and correct aspect ratio
    graph = sns.heatmap(corr, mask=mask, cmap='RdYlGn', vmax=.3, center=0, square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})
    return graph

graph_correlogram(df)

# correlation matrix with p-values
def df_correlation_matrix(df):
    numerical = df.select_dtypes(include=['int64', 'float']).columns.tolist()
    print(f'Pearson Correlation Matrix with P-Values')
    print(f'[Coef in Btm Tri / p-Values in Up Tri]')
    print(f'*** for <0.001, ** for <0.01, * for <0.05')
    print(f'-----------------------------------------')
    return df[numerical].rcorr(method='pearson').round(3)

df_correlation_matrix(df)

# dataframe missing
def df_missing_info(df):
    pd.set_option('display.max_columns', df.shape[1])
    pd.set_option('display.max_rows', df.shape[1])
    descriptive_df = pd.DataFrame()
    descriptive_df['column'] = df.columns
    descriptive_df['data type'] = df.dtypes.tolist()
    descriptive_df['# missing'] = [df[col].isnull().sum() for col in df]
    descriptive_df['% missing'] = np.round(descriptive_df['# missing'] / df.shape[0], 4)
    return descriptive_df

df_missing_info(df)

# anova df
def get_anova_df(target, df):
    # filter warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    anova_df = pd.DataFrame()
    for num in numerical_columns:
        new_row = df.anova(dv=num, between=target, detailed=False)
        new_row['Target'] = num
        new_row = new_row.rename(columns={'Target':'Feature', 'Source':'Target'})
        anova_df = pd.concat([anova_df, new_row], axis='rows')
    anova_df = anova_df[['Target', 'Feature','F', 'p-unc', 'np2']]
    return anova_df

anova_df = get_anova_df('churn', df)

anova_df.loc[
    anova_df['p-unc'] > 0.05
]

# graph of all numeric data types
def graph_numeric_histograms(df):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    numeric_columns = df.select_dtypes(include=np.number)
    for col in numeric_columns:
        sns.histplot(df[col], kde=True, color='black')
        plt.title(f'Histogram of {col.title()}')
        plt.show()
        
graph_numeric_histograms(df)

##### Mutual Information Statistic #####
# separate x & y data
X = df.drop(['churn'], axis=1)
y = LabelEncoder().fit_transform(df['churn'])

# numeric pipeline
numeric_pipeline_steps = []
numeric_pipeline_steps.append(('min-max', MinMaxScaler(feature_range=(1, 2))))
numeric_pipeline_steps.append(('box-cox', PowerTransformer(method='box-cox')))
numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

# categorical pipeline
categorical_pipeline_steps = []
categorical_pipeline_steps.append(('oe', OrdinalEncoder(dtype=np.int64)))
categorical_pipeline = Pipeline(steps=categorical_pipeline_steps)

# create transformer
cat_columns = X.select_dtypes(include=['object']).columns.tolist()
num_columns = X.select_dtypes(include=[np.number]).columns.tolist()

transformer_steps = []
transformer_steps.append(('cat', categorical_pipeline, cat_columns))
transformer_steps.append(('num', numeric_pipeline, num_columns))
preprocessing_transformer=ColumnTransformer(transformers=transformer_steps)

# preprocessing transformer
preprocessing_transformer

# preprocess X data
pp_X = preprocessing_transformer.fit_transform(X)
pp_X_columns = pd.Series(preprocessing_transformer.get_feature_names_out()).str.replace('num__|cat__', '', regex=True)
pp_X_df = pd.DataFrame(pp_X, columns=pp_X_columns)

# get discrete feature indices
discrete_features_for_mi = [ind for ind, li in enumerate(pp_X_df.columns) if li in cat_columns]

# run mutual_info_classif
mi_scores = mutual_info_classif(pp_X_df, y, discrete_features=discrete_features_for_mi, random_state=2022)

# make df of mi
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Scores': mi_scores}).sort_values('MI Scores', ascending=False)

# bar plot
plt.figure(figsize=(12, 12))
sns.barplot(y='Feature', x='MI Scores', data=mi_scores_df, color='black')
plt.title('Mutual Information Statistic')
plt.show()

# dataframe of mi scores
mi_scores_df

##### Pipeline Setup #####
# split into train, test data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('churn', axis='columns'),
    df['churn'],
    stratify=df['churn'],
    test_size=0.30,
    random_state=2022
    )

# Numeric Feature Pipeline
numeric_pipeline_steps = []
numeric_pipeline_steps.append(('scaler', StandardScaler()))
numeric_pipeline_steps.append(('poly', PolynomialFeatures(degree=2)))
numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

# Categorical Feature Pipeline
categorical_pipeline_steps = []
categorical_pipeline_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
categorical_pipeline = Pipeline(steps=categorical_pipeline_steps)

# Preprocessing Transformer
transformer_steps = []
transformer_steps.append(('cat', categorical_pipeline, make_column_selector(dtype_exclude=np.number)))
transformer_steps.append(('num', numeric_pipeline, make_column_selector(dtype_include=np.number)))
preprocessing_transformer=ColumnTransformer(transformers=transformer_steps)

# Display Transformer
set_config(display='diagram')
preprocessing_transformer

##### Models #####
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import (KNeighborsClassifier, NearestCentroid,
                               RadiusNeighborsClassifier)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Models
models = []
models.append(('DC', DummyClassifier(strategy='most_frequent')))
models.append(('AB-C', AdaBoostClassifier(n_estimators=1000, random_state=2022)))
models.append(('LR', LogisticRegression(solver='saga', penalty='elasticnet', class_weight='balanced', l1_ratio=0.5, max_iter=100_000, random_state=2022)))
models.append(('RF', RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=2022)))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier(weights='distance')))
models.append(('SVC-L', SVC(kernel='linear', class_weight='balanced')))
models.append(('SVC-P', SVC(kernel='poly', class_weight='balanced')))
models.append(('SVC-R', SVC(kernel='rbf', class_weight='balanced')))
models.append(('SVC-S', SVC(kernel='sigmoid', class_weight='balanced')))
models.append(('XGB-C', XGBClassifier(eval_metric='logloss', seed=2022)))

##### Fitting Models #####
# results dataframe
results = pd.DataFrame()

# scoring used
scoring = 'accuracy'

# CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2022)

for name, model in models:
    # Pipeline
    model_pipeline_steps = []
    model_pipeline_steps.append(('transformer', preprocessing_transformer))
    model_pipeline_steps.append(('model', model))
    pipeline = Pipeline(steps=model_pipeline_steps)
    # CV results
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    temp_df = pd.DataFrame({name: pd.Series(abs(cv_results))})
    results = pd.concat([results, temp_df], axis='columns')
    # Mean +/- std
    msg = f'{name}: {cv_results.mean().round(5)} +/- {cv_results.std().round(5)}'
    print(msg)

# Algorithm Comparison Boxplot
sns.pointplot(y='model', x=scoring,
            data=pd.melt(results, var_name='model', value_name=scoring), palette='colorblind')
plt.title('Spot Check Algorithm Boxplots')
plt.show()

# print df
results
pd.DataFrame({
    'mean':results.mean(),
    'std':results.std()
    }).sort_values('mean', ascending=True)

##### Tune Model #####
# Model 
tune_model = XGBClassifier(n_estimators=1_000, eval_metric='logloss', seed=2022)

# Scoring
scoring = 'accuracy'

# Pipeline Steps
model_pipeline_steps = []
model_pipeline_steps.append(('transformer', preprocessing_transformer))
model_pipeline_steps.append(('model', tune_model))
model_pipeline = Pipeline(steps=model_pipeline_steps)

# Param Grid
tune_model.get_params()
model_pipeline.get_params()
param_grid = {
    'model__learning_rate': np.linspace(0.01, 0.2, 50),
    'model__max_depth': np.arange(2, 11, 1, dtype=int),
    'model__subsample': np.linspace(0.5, 1.0, 50),
    'model__colsample_bytree': np.linspace(0.5, 1.0, 50),
    }

# BayesSearchCV
bs_model = BayesSearchCV(
    estimator=model_pipeline,
    search_spaces=param_grid,
    scoring=scoring,
	n_iter=30,
    n_jobs=-1,
    random_state=2022)

# Display Model
bs_model

# Fit Model on Train
bs_model.fit(X_train, y_train)

# best score
print(f'best {scoring} score: {abs(bs_model.best_score_).round(5)}')

# best params
best_param_df = pd.DataFrame(bs_model.best_params_.items(), columns=['Parameter', 'Value'])
best_param_df['Parameter'] = best_param_df['Parameter'].str.replace('model__', '')
best_param_df['Value'] = np.round(best_param_df['Value'], 4)
best_param_df

# best estimator
best_model_pipeline = bs_model.best_estimator_

###### ROC-AUC #####
# ROC AUC on Test Set
RocCurveDisplay.from_estimator(
    estimator=best_model_pipeline,
    X=X_test,
    y=y_test
    )
plt.show()

##### Confusion Matrix #####
# Confusion on Test Set
ConfusionMatrixDisplay.from_estimator(
    estimator=best_model_pipeline,
    X=X_test,
    y=y_test,
    cmap='Blues',
    colorbar=False)
plt.suptitle('Confusing Matrix')
plt.title(f'Accuracy of: {accuracy_score(y_true=y_test, y_pred=best_model_pipeline.predict(X_test))}')
plt.show()

##### Finalization #####
# Save Model
model_name = 'XGB-C'
model_file_name = f'models/{model_name} on {datetime.now().strftime("%Y %b %d at %H.%M.%S")}.pkl'
joblib.dump(best_model_pipeline, model_file_name)