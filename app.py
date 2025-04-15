import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import io
import time

st.set_page_config(page_title="Machine Learning Dashboard", layout="centered")
st.title("Machine Learning Dashboard")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Section :", ["Importer un Fichier", "Analyse EDA", "Modélisation ML", "Prédictions"])

dataset = st.session_state.get("dataset", None)
trained_models = st.session_state.get("trained_models", None)
model_options = st.session_state.get("model_options", None)
features_df = st.session_state.get("features_df", None)

if menu == "Importer un Fichier":
    st.subheader("Chargement du fichier CSV")
    file = st.file_uploader("Sélectionnez un fichier .csv", type="csv")

    if file:
        with st.spinner("Lecture du fichier en cours..."):
            try:
                time.sleep(1)
                dataset = pd.read_csv(file, on_bad_lines='skip')
                st.session_state["dataset"] = dataset
                st.success("Fichier chargé avec succès")
                st.dataframe(dataset.head(), use_container_width=True)
            except Exception as err:
                st.error(f"Erreur de chargement : {err}")

elif menu == "Analyse EDA":
    st.subheader("Analyse exploratoire des données")
    if dataset is not None:
        tab_stats, tab_graphs, tab_nulls = st.tabs(["Statistiques", "Graphiques", "Valeurs Manquantes"])

        with tab_stats:
            st.markdown("**Résumé Statistique**")
            st.dataframe(dataset.describe(), use_container_width=True)
            st.markdown("**Structure des Données :**")
            buf = io.StringIO()
            dataset.info(buf=buf)
            st.text(buf.getvalue())

        with tab_graphs:
            st.markdown("**Visualisation de colonnes numériques**")
            num_col = st.selectbox("Sélectionnez une colonne :", dataset.select_dtypes(include=['number']).columns)
            if num_col:
                fig = px.histogram(dataset, x=num_col)
                st.plotly_chart(fig, use_container_width=True)

        with tab_nulls:
            st.markdown("**Valeurs Manquantes**")
            nulls = dataset.isnull().sum()
            if nulls.sum() > 0:
                st.write(nulls)
                if st.button("Remplir les valeurs nulles par la moyenne"):
                    dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
                    st.session_state["dataset"] = dataset
                    st.success("Valeurs manquantes remplacées")
            else:
                st.info("Aucune valeur manquante détectée.")
    else:
        st.warning("Veuillez d'abord importer un fichier.")

elif menu == "Modélisation ML":
    st.subheader("Modélisation Machine Learning")
    if dataset is not None:
        st.sidebar.subheader("Configuration")
        task_type = st.sidebar.selectbox("Tâche :", ["Classification", "Régression"])
        target = st.sidebar.selectbox("Variable cible :", dataset.columns)

        if target:
            features = dataset.drop(columns=[target])
            labels = dataset[target]
            st.session_state["features_df"] = features

            cat_columns = features.select_dtypes(include=['object']).columns
            if len(cat_columns) > 0:
                encoder = OneHotEncoder()
                enc_data = pd.DataFrame(encoder.fit_transform(features[cat_columns]).toarray())
                features = features.drop(columns=cat_columns).reset_index(drop=True).join(enc_data)

            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

            if task_type == "Classification":
                model_options = {
                    "Forêt Aléatoire": RandomForestClassifier(),
                    "Boosting": GradientBoostingClassifier(),
                    "Régression Logistique": LogisticRegression(max_iter=1000),
                    "SVC": SVC()
                }
            else:
                model_options = {
                    "Forêt Aléatoire (Reg)": RandomForestRegressor(),
                    "Boosting (Reg)": GradientBoostingRegressor(),
                    "Régression Linéaire": LinearRegression(),
                    "SVR": SVR()
                }

            st.session_state["model_options"] = model_options

            performance = []
            with st.spinner("Entraînement des modèles..."):
                for mdl_name, mdl in model_options.items():
                    mdl.fit(X_train, y_train)
                    y_pred = mdl.predict(X_test)

                    if task_type == "Classification":
                        rep = classification_report(y_test, y_pred, output_dict=True)
                        performance.append({
                            "Modèle": mdl_name,
                            "Précision": rep['weighted avg']['precision'],
                            "Rappel": rep['weighted avg']['recall'],
                            "F1-Score": rep['weighted avg']['f1-score']
                        })
                    else:
                        performance.append({
                            "Modèle": mdl_name,
                            "MAE": mean_absolute_error(y_test, y_pred),
                            "MSE": mean_squared_error(y_test, y_pred),
                            "R²": r2_score(y_test, y_pred)
                        })

                st.session_state["trained_models"] = model_options
                st.success("Modèles entraînés avec succès")
                st.dataframe(pd.DataFrame(performance), use_container_width=True)
    else:
        st.error("Aucun fichier n'a été chargé.")

elif menu == "Prédictions":
    st.subheader("Prédictions en direct")

    if not trained_models and model_options:
        trained_models = model_options
        st.session_state["trained_models"] = trained_models

    if not trained_models:
        st.info("Aucun modèle disponible. Veuillez d'abord entraîner un modèle.")
    elif features_df is None:
        st.info("Les données d'entrée ne sont pas disponibles.")
    else:
        model_choice = st.selectbox("Choisir un modèle :", list(model_options.keys()))

        if model_choice in trained_models:
            chosen_model = trained_models[model_choice]

            inputs_dict = {}
            for column in features_df.columns:
                inputs_dict[column] = st.slider(
                    f"{column}", 
                    float(features_df[column].min()), 
                    float(features_df[column].max()), 
                    float(features_df[column].mean())
                )

            prediction_input = pd.DataFrame([inputs_dict])

            if st.button("Prédire"):
                with st.spinner("Prédiction en cours..."):
                    result = chosen_model.predict(prediction_input)
                    st.success(f"Résultat de la prédiction : {result[0]}")
        else:
            st.error("Le modèle sélectionné est introuvable.")
