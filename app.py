import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.title("Домашняя работа 1")

df_train = pd.read_csv("df_train_final.csv")
num_cols = df_train.select_dtypes(["int", "float"]).columns

st.header("Визуализация обучающего набора данных")

columns = st.multiselect("Выберите колонки для визуализации", num_cols)

if columns:

    st.subheader("Парные диаграммы")

    pairplot_fig = sns.pairplot(df_train[columns])
    st.pyplot(pairplot_fig.figure)

    st.subheader("Корреляционная матрица")

    fig, ax = plt.subplots()
    sns.heatmap(df_train[columns].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.header("Визуализация весов обученной модели")

with open('ridge_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

model_coefs = pipeline.named_steps['model'].coef_
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

fig, ax = plt.subplots()
sns.barplot(x=feature_names, y=model_coefs, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

st.header("Загрузка данных для получения предсказаний")

uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    try: 
        preds = pipeline.predict(data)
        st.write("Предсказания обученной модели:")
        st.dataframe(preds)
    except:
        st.write("Ошибка при обработке данных")