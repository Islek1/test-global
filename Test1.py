import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # Замените на ваш классификатор
import matplotlib.pyplot as plt


# Загрузка данных
data = pd.read_csv("https://biconsult.ru/img/datascience-ml-ai/student-mat.csv")

