import mystreamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import os

# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Depósito Bancario",
    layout="centered",
)

st.title("Predicción de Subscripción a Depósito Bancario")
st.markdown(
    "Introduce los datos del cliente para predecir si contratará un depósito a plazo."
)

# ── Carga del modelo ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model.pkl")

@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pkl.load(f)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"No se encontró el modelo en `{MODEL_PATH}`. "
        "Asegúrate de que `final_model.pkl` esté en el mismo directorio que este script."
    )
    st.stop()

# ── Formulario de entrada ────────────────────────────────────────────────────
st.subheader("Datos del cliente")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad (age)", min_value=18, max_value=100, value=40)

    job = st.selectbox("Tipo de trabajo (job)", [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ])

    marital = st.selectbox("Estado civil (marital)", ["married", "single", "divorced"])

    education = st.selectbox("Nivel de educación (education)", [
        "primary", "secondary", "tertiary", "unknown"
    ])

    default = st.selectbox("¿Tiene crédito impagado? (default)", ["no", "yes"])

    balance = st.number_input("Balance anual medio (balance) €", value=1000, step=100)

    housing = st.selectbox("¿Tiene hipoteca? (housing)", ["no", "yes"])

    loan = st.selectbox("¿Tiene préstamo personal? (loan)", ["no", "yes"])

with col2:
    contact = st.selectbox("Tipo de contacto (contact)", [
        "cellular", "telephone", "unknown"
    ])

    day = st.number_input("Último día de contacto (day)", min_value=1, max_value=31, value=15)

    month = st.selectbox("Último mes de contacto (month)", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])

    duration = st.number_input(
        "Duración del último contacto en segundos (duration)", min_value=0, value=200
    )

    campaign = st.number_input(
        "Número de contactos en esta campaña (campaign)", min_value=1, value=1
    )

    pdays = st.number_input(
        "Días desde último contacto previo (pdays) — usa -1 si no hubo contacto",
        min_value=-1, value=-1
    )

    previous = st.number_input(
        "Contactos antes de esta campaña (previous)", min_value=0, value=0
    )

    poutcome = st.selectbox("Resultado campaña anterior (poutcome)", [
        "unknown", "failure", "other", "success"
    ])

# ── Construcción del DataFrame de entrada ────────────────────────────────────
input_data = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
}])

# ── Predicción ───────────────────────────────────────────────────────────────
st.subheader("Resultado de la predicción")

if st.button("Predecir", use_container_width=True):
    prediction = model.predict(input_data)[0]

    # Probabilidad si el modelo la soporta
    prob_text = ""
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, prob))
        prob_yes = prob_dict.get("yes", prob_dict.get(1, None))
        if prob_yes is not None:
            prob_text = f"  \nProbabilidad de **SÍ** suscribir: **{prob_yes:.1%}**"

    if str(prediction).lower() in ("yes", "1", "true"):
        st.success(f"El cliente **SÍ** contratará el depósito.{prob_text}")
    else:
        st.error(f"El cliente **NO** contratará el depósito.{prob_text}")

    # Mostrar los datos introducidos
    with st.expander("Ver datos introducidos"):
        st.dataframe(input_data.T.rename(columns={0: "Valor"}))

# ── Sección de verificación — dos instancias fijas ───────────────────────────
st.markdown("---")
st.subheader("🧪 Verificación con instancias fijas")
st.markdown(
    "Comprueba que las predicciones de la pipeline coinciden con las de la app. "
    "Estas dos instancias deben dar el mismo resultado que el notebook."
)

instancia_1 = pd.DataFrame([{
    "age": 45, "job": "management", "marital": "married", "education": "tertiary",
    "default": "no", "balance": 5000, "housing": "yes", "loan": "no",
    "contact": "cellular", "day": 10, "month": "may", "duration": 800,
    "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"
}])

instancia_2 = pd.DataFrame([{
    "age": 30, "job": "blue-collar", "marital": "single", "education": "primary",
    "default": "no", "balance": -200, "housing": "yes", "loan": "yes",
    "contact": "telephone", "day": 20, "month": "nov", "duration": 90,
    "campaign": 5, "pdays": -1, "previous": 0, "poutcome": "failure"
}])

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Instancia 1** (cliente favorable)")
    pred1 = model.predict(instancia_1)[0]
    st.info(f"Predicción: **{pred1}**")
    st.dataframe(instancia_1.T.rename(columns={0: "Valor"}))

with col_b:
    st.markdown("**Instancia 2** (cliente desfavorable)")
    pred2 = model.predict(instancia_2)[0]
    st.info(f"Predicción: **{pred2}**")
    st.dataframe(instancia_2.T.rename(columns={0: "Valor"}))

st.caption("Práctica 1 de Aprendizaje Automático")