import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

st.title("Домашняя работа 1")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "ridge_pipeline.pkl"
DF_TRAIN_PATH = MODEL_DIR / "X_train_cat.csv"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    df_train = pd.read_csv(DF_TRAIN_PATH)
except Exception as e:
    st.error(f"Ошибка загрузки набора данных: {e}")
    st.stop()

cat_cols = df_train.select_dtypes(["object"]).columns
num_cols = df_train.select_dtypes(["int", "float"]).columns

st.header("Визуализация обучающего набора данных")

columns = st.multiselect("Выберите **числовые** признаки для визуализации", num_cols)

if columns:
    try:
        st.subheader("Парные диаграммы")

        pairplot_fig = sns.pairplot(df_train[columns])
        st.pyplot(pairplot_fig.figure)

        st.subheader("Корреляционная матрица")

        fig, ax = plt.subplots()
        sns.heatmap(df_train[columns].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    except Exception as e: 
        st.error(f"Ошибка визуализации обучающего набора данных: {e}")
        st.stop()

st.header("Визуализация весов обученной модели")

try: 
    model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

model_coefs = model.named_steps['model'].coef_
feature_names = model.named_steps['preprocessor'].get_feature_names_out()

try: 
    fig, ax = plt.subplots()
    sns.barplot(x=feature_names, y=model_coefs, ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)
except Exception as e: 
    st.error(f"Ошибка визуализации весов модели: {e}")
    st.stop()

st.header("Загрузка данных для получения предсказаний")

uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

if uploaded_file is not None:
    try: 
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка при обработке CSV-файлa {e}")
        st.stop()
    
    try: 
        preds = model.predict(data)
    except Exception as e:
        st.error(f"Ошибка при обработке данных: {e}")
        st.stop()
    
    st.subheader("Результаты")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Всего наблюдений", len(data))
    with col2:
        avg_pred = preds.mean()
        st.metric("Средняя стоимость автомобиля", f"{avg_pred:.1f}")
    st.write("Предсказания модели:")
    st.dataframe(preds)

    st.write("Распределение предсказаний модели:")
    fig, ax = plt.subplots()
    sns.histplot(preds, ax=ax)
    st.pyplot(fig)
    


st.header("Сделать предсказание для одного наблюдения")

with st.form("prediction_form"):
    col_left, col_right = st.columns(2)
    input_data = {}
    
    with col_left:
        st.write("**Категориальные:**")
        for col in cat_cols:
            unique_vals = sorted(df_train[col].astype(str).unique().tolist())
            input_data[col] = st.selectbox(col, unique_vals, key=f"cat_{col}")

    with col_right:
        st.write("**Числовые:**")
        for col in num_cols:
            val = float(df_train[col].median())
            input_data[col] = st.number_input(col, value=val, key=f"num_{col}")

    submitted = st.form_submit_button("Предсказать", use_container_width=True)

if submitted:
    try:
        data = pd.DataFrame([input_data])
        pred = model.predict(data)[0]

        st.success(f"**Стоимость автомобиля автомобиля:** {pred}")
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")