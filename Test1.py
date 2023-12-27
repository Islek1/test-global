import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # Замените на ваш классификатор
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Загрузка данных
data = pd.read_csv("https://biconsult.ru/img/datascience-ml-ai/student-mat.csv")
numeric_columns = data.select_dtypes(include=['number'])
numeric_columns = data.select_dtypes(include=['int64', 'float64'])
X = numeric_columns.drop(columns=['G3'])
y = data['G3']
g

