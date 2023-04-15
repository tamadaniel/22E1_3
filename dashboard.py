import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd

# Title
st.title("Model Experimentation with MLflow")

# Configuração da URL do servidor MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Substitua pela URL do seu servidor MLflow

# ID do experimento desejado
experiment_id = "501323974373270564"

# Obtém as informações do experimento
runs = mlflow.search_runs(experiment_id)

if not runs.empty:
    # Exibe a lista de colunas para seleção
    column_names = runs.columns.tolist()
    selected_column_name = st.selectbox("Selecione uma coluna", column_names)

    if selected_column_name in column_names:
        # Obtém os valores da coluna selecionada
        column_values = runs[selected_column_name]

        # Exibe os valores da coluna selecionada
        st.write(f"Valores da Coluna '{selected_column_name}':")
        st.write(column_values)
    else:
        st.write(f"Nenhuma coluna encontrada com o nome '{selected_column_name}'.")
else:
    st.write(f"Nenhum run encontrado no experimento de ID '{experiment_id}'.")
