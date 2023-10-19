# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os

os.chdir('/Users/ahmedbastawissy/Desktop/Projekte/Airbnb_data')


# Datensatz laden
airbnb_data = pd.read_csv('Airbnb_Open_Data.csv')

# Ersetzen von Leerzeichen durch Unterstriche für alle Spaltennamen
airbnb_data.columns = [col.replace(' ', '_') for col in airbnb_data.columns]


# Nur ausgewählte Variablen behalten
selected_columns = ['id', 'host_identity_verified', 'neighbourhood', 'number_of_reviews', 'instant_bookable', 
                    'room_type', 'availability_365', 'minimum_nights', 'reviews_per_month', 'price', 
                    'service_fee', 'cancellation_policy']


airbnb_data = airbnb_data[selected_columns]



# Überprüfen der Datentypen aller Spalten
print(airbnb_data.dtypes)


# Entfernen von Währungszeichen und Umwandlung in float
airbnb_data['price'] = airbnb_data['price'].str.replace('$', '').str.replace(',', '').astype(float)
airbnb_data['service_fee'] = airbnb_data['service_fee'].str.replace('$', '').str.replace(',', '').astype(float)

# Umwandlung in category
airbnb_data['cancellation_policy'] = airbnb_data['cancellation_policy'].astype('category')
airbnb_data['neighbourhood'] = airbnb_data['neighbourhood'].astype('category')
airbnb_data['room_type'] = airbnb_data['room_type'].astype('category')
airbnb_data['instant_bookable'] = airbnb_data['instant_bookable'].astype('category')
airbnb_data['host_identity_verified'] = airbnb_data['host_identity_verified'].astype('category')


# Univariate Analyse:
# Importieren der benötigten Bibliotheken
import matplotlib.pyplot as plt
import seaborn as sns

# Histogramme für numerische Variablen
numerical_columns = ['number_of_reviews', 'availability_365', 'minimum_nights', 'reviews_per_month', 'price', 'service_fee']

# Datenbereinigung: Entfernen von Zeilen mit NaN oder Inf in den relevanten Spalten
airbnb_data = airbnb_data.dropna(subset=numerical_columns)


for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(airbnb_data[column], bins=30, kde=True)
    plt.title(f'Histogram of {column}')
    plt.show()



# Mittelwerte, Median, etc. für numerische Variablen
pd.set_option('display.max_columns', None)
print("Statistical Summary of Numerical Columns:")
print(airbnb_data[numerical_columns].describe())

# Barplots für kategoriale Variablen
categorical_columns = ['host_identity_verified', 'instant_bookable', 'room_type', 'cancellation_policy']

for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=airbnb_data, y=column, order = airbnb_data[column].value_counts().index)
    plt.title(f'Barplot of {column}')
    plt.show()


# Stark ausgebucht oder nicht
airbnb_data['highly_booked'] = airbnb_data['availability_365'] < 60


sns.countplot(data=airbnb_data, x='highly_booked')
plt.title('Highly Booked Properties')
plt.show()


# Outlier-Analyse
import pandas as pd
import numpy as np
from scipy.stats import zscore

# Z-Score Methode zur Identifikation von Ausreißern
def identify_outliers_zscore(data, threshold=3):
    z_scores = zscore(data)
    outliers = np.abs(z_scores) > threshold
    return outliers

# IQR Methode zur Identifikation von Ausreißern
def identify_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers


for column in numerical_columns:
    
    # Identifiziere Ausreißer mit der Z-Score Methode
    outliers_zscore = identify_outliers_zscore(airbnb_data[column])
    
    # Identifiziere Ausreißer mit der IQR Methode
    outliers_iqr = identify_outliers_iqr(airbnb_data[column])
    
    # Finde Datenpunkte, die in BEIDEN Methoden als Ausreißer identifiziert wurden
    combined_outliers = outliers_zscore & outliers_iqr
    
    # Entferne diese Ausreißer aus dem Datensatz
    airbnb_data = airbnb_data[~combined_outliers]

# Der Datensatz 'airbnb_data' enthält nun keine der identifizierten kombinierten Ausreißer mehr



# Bivariate Analyse

# Boxplots für kategoriale Variablen vs. Verfügbarkeit
# Bivariate Analyse ohne Ausreißer

# Boxplots für kategoriale Variablen vs. Verfügbarkeit
for column in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=column, y='availability 365', data=airbnb_data, showfliers=False)
    plt.title(f'Boxplot of {column} vs Availability (Without Outliers)')
    plt.xticks(rotation=0)
    plt.show()

# Korrelation und Scatterplots für numerische Variablen vs. Verfügbarkeit unter Entfernung von Ausreißern

for column in numerical_columns:
    Q1 = airbnb_data[column].quantile(0.25)
    Q3 = airbnb_data[column].quantile(0.75)
    IQR = Q3 - Q1
    filtered_data = airbnb_data[(airbnb_data[column] >= Q1 - 1.5*IQR) & (airbnb_data[column] <= Q3 + 1.5*IQR)]
    
    correlation = filtered_data[column].corr(filtered_data['availability_365'])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=filtered_data[column], y=filtered_data['availability_365'])
    plt.title(f'Scatterplot of {column} vs Availability (Correlation: {correlation:.2f})')
    plt.show()

# Multivariate Analyse

from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# Erstellen einer Matrix der Korrelation
correlation_matrix = airbnb_data[numerical_columns].corr()

# Leeres Array für die p-Werte
p_values = np.empty(correlation_matrix.shape)

# p-Werte berechnen
for row in correlation_matrix.index:
    for col in correlation_matrix.columns:
        if row == col:
            p_values[correlation_matrix.index.get_loc(row), correlation_matrix.columns.get_loc(col)] = np.nan
        else:
            _, p_value = pearsonr(airbnb_data[row], airbnb_data[col])
            p_values[correlation_matrix.index.get_loc(row), correlation_matrix.columns.get_loc(col)] = p_value

# Signifikanzlabels basierend auf den p-Werten
significance_labels = correlation_matrix.applymap(lambda x: "")
for (i, j), p_value in np.ndenumerate(p_values):
    if not np.isnan(p_value):
        if p_value < 0.001:
            significance_labels.iloc[i, j] = "***"
        elif p_value < 0.01:
            significance_labels.iloc[i, j] = "**"
        elif p_value < 0.05:
            significance_labels.iloc[i, j] = "*"

# Heatmap der Korrelation mit annot als Kombination von Korrelation und Signifikanz


# Kombinierte Annotation aus Korrelationswerten und Signifikanzlabels
annot = correlation_matrix.applymap('{:.2f}'.format) + significance_labels

# Einstellen des Hintergrundstils für bessere Lesbarkeit
sns.set_style("whitegrid")

# Erstellen der Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', vmin=-1, vmax=1, fmt="s", linewidths=0.5, linecolor='white')

# Verbesserte Achsentitel und -beschriftungen für bessere Lesbarkeit
plt.title('Korrelations-Heatmap', fontsize=16)

merkmal_namen = [
    "Anzahl der Bewertungen", 
    "Verfügbarkeit (365 Tage)", 
    "Mindestübernachtungen", 
    "Bewertungen pro Monat", 
    "Preis", 
    "Servicegebühr"
]
ax = plt.gca()
ax.set_xticklabels(merkmal_namen, fontsize=12, horizontalalignment='center', rotation=45)
ax.set_yticklabels(merkmal_namen, fontsize=12, verticalalignment='center', rotation=45)
plt.tight_layout()

# Anzeigen der Heatmap
plt.show()


# Statistische Tests
import scipy.stats as stats

# T-Test
# Gruppen erstellen
group_verified = airbnb_data[airbnb_data['host_identity_verified'] == 'verified']['availability_365']
group_not_verified = airbnb_data[airbnb_data['host_identity_verified'] == 'unconfirmed']['availability_365']

# Unabhängigen T-Test durchführen für 'host_identity_verified'
t_stat, p_value = stats.ttest_ind(group_verified, group_not_verified)

print(f"T-Statistik: {t_stat:.3f}")
print(f"P-Wert: {p_value:.3f}")
2
# Entscheidung basierend auf dem p-Wert
alpha = 0.05  # Signifikanzniveau
if p_value < alpha:
    print("Es gibt einen signifikanten Unterschied im Durchschnitt der Verfügbarkeit zwischen den beiden Gruppen.")
else:
    print("Es gibt keinen signifikanten Unterschied im Durchschnitt der Verfügbarkeit zwischen den beiden Gruppen.")

# ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ANOVA für 'room type'
model_room_type = ols('availability_365 ~ C(room_type)', data=airbnb_data).fit()
anova_table_room_type = sm.stats.anova_lm(model_room_type, typ=2)
print("ANOVA-Test für room type:")
print(anova_table_room_type)
print("\n")

# ANOVA für 'cancellation_policy'
model_cancellation_policy = ols('availability_365 ~ C(cancellation_policy)', data=airbnb_data).fit()
anova_table_cancellation_policy = sm.stats.anova_lm(model_cancellation_policy, typ=2)
print("ANOVA-Test für cancellation_policy:")
print(anova_table_cancellation_policy)


# Ml-Modell

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
  
# Daten vorbereiten
X = airbnb_data[['room_type', 'reviews_per_month', 'minimum_nights', 'price']]
y = airbnb_data['availability_365']

def prepare_data(df):
    df = df.copy()
    
    # 'room_type' in numerische Werte konvertieren
    label_encoder = LabelEncoder()
    df['room_type'] = label_encoder.fit_transform(df['room_type'])
    
    return df, label_encoder.classes_

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Random Forest Modell
    rf_model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 500]
    }
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                               scoring='neg_mean_squared_error', cv=4, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_n_estimators = grid_search.best_params_['n_estimators']
    print(f"Der optimale Wert für n_estimators ist: {best_n_estimators}")

    # Modell mit den besten Hyperparametern trainieren:
    rf_model = RandomForestRegressor(n_estimators=best_n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)

    # Vorhersagen machen
    y_pred = rf_model.predict(X_test)

    # Modell bewerten
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = rf_model.score(X_test, y_test)
    
    results = {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Feature Importance": rf_model.feature_importances_
    }
    
    return results


X, classes = prepare_data(airbnb_data[['room_type', 'reviews_per_month', 'minimum_nights', 'price']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = train_and_evaluate(X_train, X_test, y_train, y_test)

print(f"Mean Squared Error: {results['MSE']}")
print(f"Root Mean Squared Error: {results['RMSE']}")
print(f"R^2 Score: {results['R2']}")

# Feature-Wichtigkeiten anzeigen
for feature, importance in zip(X.columns, results['Feature Importance']):
    print(f"{feature}: {importance:.3f}")


