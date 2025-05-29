import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

try:
    data = pd.read_csv('games.csv')
except FileNotFoundError:
    st.error("Файл 'games.csv' не найден.")
    st.stop()

data = data[data['winner'] != 'draw']
data['winner'] = data['winner'].map({'white': 0, 'black': 1})
data['rating_difference'] = data['white_rating'] - data['black_rating']

features = ['white_rating', 'black_rating', 'turns', 'rated', 'opening_ply', 'rating_difference']
target = 'winner'
X = data[features].copy()
y = data[target]
X.loc[:, 'rated'] = X['rated'].astype(int)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки (только для примера, можно обучить на всех данных)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_learning_rate = 0.2
best_max_depth = 3
best_min_samples_split = 2
best_min_samples_leaf = 6
best_subsample = 1.0

# Streamlit UI
st.title("Предсказание результата шахматной партии")

# Входные данные от пользователя
white_rating = st.slider("Рейтинг белых", 800, 2800, 1500)
black_rating = st.slider("Рейтинг черных", 800, 2800, 1500)
turns = st.slider("Количество ходов", 1, 300, 50)
rated = st.selectbox("Рейтинговая игра?", [0, 1])
opening_ply = st.slider("Количество ходов в дебюте", 0, 30, 5)

# Возможность изменять n_estimators
n_estimators = st.slider("Количество деревьев (n_estimators)", 50, 500, 400)

# Переобучение модели при изменении n_estimators
model = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=best_learning_rate,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    subsample=best_subsample,
    random_state=42
)
model.fit(X_train, y_train)

# Создание DataFrame с введенными значениями
input_data = pd.DataFrame({
    'white_rating': [white_rating],
    'black_rating': [black_rating],
    'turns': [turns],
    'rated': [rated],
    'opening_ply': [opening_ply],
    'rating_difference': [white_rating - black_rating]
})

# Масштабирование введенных данных
input_scaled = scaler.transform(input_data)

# Предсказание результата
prediction = model.predict(input_scaled)[0]

# Выводим результат
if prediction == 0:
    st.write("Победитель: Белые")
elif prediction == 1:
    st.write("Победитель: Черные")