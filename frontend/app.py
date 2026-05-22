import os
from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Olist — Prédiction livraison",
    page_icon="📦",
)

st.title("📦 Prédiction du temps de livraison")
st.caption("Olist Brazilian E-Commerce — RandomForestRegressor")

st.sidebar.header("Paramètres de la commande")

price = st.sidebar.number_input(
    "Prix total des articles",
    min_value=1.0,
    value=89.90,
    step=10.0,
)

freight_value = st.sidebar.number_input(
    "Frais de port",
    min_value=0.0,
    value=12.50,
    step=1.0,
)

payment_installments = st.sidebar.slider(
    "Nombre de versements",
    min_value=1,
    max_value=12,
    value=1,
)

payment_value = st.sidebar.number_input(
    "Montant total payé",
    min_value=1.0,
    value=102.40,
    step=10.0,
)

order_item_id = st.sidebar.slider(
    "Nombre d'articles",
    min_value=1,
    max_value=10,
    value=1,
)

purchase_date = st.sidebar.date_input(
    "Date d'achat",
    value=datetime(2018, 6, 10),
)

seller_state = st.sidebar.selectbox(
    "État du vendeur",
    ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "ES", "CE"],
)

customer_state = st.sidebar.selectbox(
    "État du client",
    ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "ES", "CE"],
)

if st.button("🔮 Prédire le temps de livraison", type="primary"):
    payload = {
        "price": price,
        "freight_value": freight_value,
        "payment_installments": payment_installments,
        "payment_value": payment_value,
        "order_item_id": order_item_id,
        "order_purchase_timestamp": datetime.combine(
            purchase_date,
            datetime.min.time(),
        ).isoformat(),
        "seller_state": seller_state,
        "customer_state": customer_state,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()

        result = response.json()
        days = result["delivery_time_days"]

        st.success(f"⏱ Temps de livraison prédit : **{days:.1f} jours**")

        if days < 7:
            st.info("Livraison rapide — client probablement satisfait.")
        elif days < 15:
            st.warning("Livraison moyenne — délai acceptable mais à surveiller.")
        else:
            st.error("Livraison longue — risque d'insatisfaction client.")

        with st.expander("Voir le payload envoyé à l'API"):
            st.json(payload)

    except requests.RequestException as error:
        st.error(f"Erreur lors de l'appel API : {error}")

with st.expander("ℹ️ À propos du modèle"):
    st.markdown("""
        **Algorithme** : RandomForestRegressor
        **Objectif** : prédire le temps de livraison en jours
        **Données** : Olist Brazilian E-Commerce
        **Features utilisées** : prix, frais de port, paiement, nombre d'articles,
        date d'achat, état vendeur et état client
        **Modèle** : stocké dans MinIO et chargé par l'API FastAPI
        """)
