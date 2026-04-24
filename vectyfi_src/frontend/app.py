import streamlit as st
import requests
import random

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(page_title="Vectyfi · Tender Prediction", layout="centered")

API_URL = "https://vectyfi-api-828368828432.europe-west1.run.app/predict"

# ── Valeurs possibles pour les champs catégoriels ────────────────────────────
TOP_TYPES        = ["OPEN", "RESTRICTED", "NEGOTIATED", "OTHER"]
COUNTRIES        = ["FR", "DE", "IT", "ES", "BE", "NL", "PL", "PT", "SE", "AT"]
CONTRACT_TYPES   = ["WORKS", "SUPPLIES", "SERVICES"]
CAE_TYPES        = ["OTH", "RA", "LA", "PUB", "DEF", "CGA", "HEA", "EU", "REG"]
MAIN_ACTIVITIES  = ["COSAL", "HEALTH", "DEFENCE", "EDUCATION", "RCREA", "HOUCOM",
                    "ENVIR", "ECONEC", "SEWAGE", "GENECO", "RAILSV", "AIRPORT",
                    "PORTFA", "GASELE", "HEATCO", "WATER", "OTHER"]

# ── Données aléatoires pour le bouton "Random" ───────────────────────────────
def random_input():
    return {
        "B_MULTIPLE_CAE":  random.choice(["Y", "N"]),
        "B_EU_FUNDS":      random.choice(["Y", "N"]),
        "B_GPA":           random.choice(["Y", "N"]),
        "B_FRA_AGREEMENT": random.choice(["Y", "N"]),
        "B_ACCELERATED":   random.choice(["Y", "N"]),
        "LOTS_NUMBER":       float(random.randint(1, 20)),
        "YEAR":              float(random.randint(2015, 2023)),
        "CRIT_PRICE_WEIGHT": float(random.randint(20, 80)),
        "CRIT_CODE":         float(random.randint(1, 5)),
        "TOP_TYPE":          random.choice(TOP_TYPES),
        "ISO_COUNTRY_CODE":  random.choice(COUNTRIES),
        "TYPE_OF_CONTRACT":  random.choice(CONTRACT_TYPES),
        "CAE_TYPE":          random.choice(CAE_TYPES),
        "MAIN_ACTIVITY":     random.choice(MAIN_ACTIVITIES),
    }

# ── Initialise le session state avec des valeurs par défaut ──────────────────
if "inputs" not in st.session_state:
    st.session_state.inputs = random_input()

# ── Titre ─────────────────────────────────────────────────────────────────────
st.title("Vectyfi — Tender Prediction")
st.caption("Remplis les champs ou génère des données aléatoires, puis appelle l'API.")

# ── Bouton Random ─────────────────────────────────────────────────────────────
if st.button("🎲 Générer des données aléatoires"):
    st.session_state.inputs = random_input()

st.divider()

# ── Formulaire ────────────────────────────────────────────────────────────────
inp = st.session_state.inputs

st.subheader("Champs binaires")
col1, col2 = st.columns(2)
with col1:
    b_multiple_cae  = st.selectbox("Multiple CAE",  ["Y", "N"], index=["Y","N"].index(inp["B_MULTIPLE_CAE"]))
    b_eu_funds      = st.selectbox("EU Funds",      ["Y", "N"], index=["Y","N"].index(inp["B_EU_FUNDS"]))
    b_gpa           = st.selectbox("GPA",           ["Y", "N"], index=["Y","N"].index(inp["B_GPA"]))
with col2:
    b_fra_agreement = st.selectbox("FRA Agreement", ["Y", "N"], index=["Y","N"].index(inp["B_FRA_AGREEMENT"]))
    b_accelerated   = st.selectbox("Accelerated",   ["Y", "N"], index=["Y","N"].index(inp["B_ACCELERATED"]))

st.subheader("Champs numériques")
col3, col4 = st.columns(2)
with col3:
    lots_number       = st.number_input("Lots Number",       value=inp["LOTS_NUMBER"],       min_value=1.0)
    year              = st.number_input("Year",              value=inp["YEAR"],              min_value=2000.0, max_value=2030.0, step=1.0)
with col4:
    crit_price_weight = st.number_input("Crit Price Weight", value=inp["CRIT_PRICE_WEIGHT"], min_value=0.0, max_value=100.0)
    crit_code         = st.number_input("Crit Code",         value=inp["CRIT_CODE"],         min_value=1.0)

st.subheader("Champs catégoriels")
col5, col6 = st.columns(2)
with col5:
    top_type         = st.selectbox("Top Type",          TOP_TYPES,       index=TOP_TYPES.index(inp["TOP_TYPE"]))
    iso_country_code = st.selectbox("Country Code",      COUNTRIES,       index=COUNTRIES.index(inp["ISO_COUNTRY_CODE"]))
    type_of_contract = st.selectbox("Type of Contract",  CONTRACT_TYPES,  index=CONTRACT_TYPES.index(inp["TYPE_OF_CONTRACT"]))
with col6:
    cae_type         = st.selectbox("CAE Type",          CAE_TYPES,       index=CAE_TYPES.index(inp["CAE_TYPE"]))
    main_activity    = st.selectbox("Main Activity",     MAIN_ACTIVITIES, index=MAIN_ACTIVITIES.index(inp["MAIN_ACTIVITY"]))

st.divider()

# ── Appel API ─────────────────────────────────────────────────────────────────
if st.button("🚀 Prédire", type="primary"):
    payload = {
        "B_MULTIPLE_CAE":  b_multiple_cae,
        "B_EU_FUNDS":      b_eu_funds,
        "B_GPA":           b_gpa,
        "B_FRA_AGREEMENT": b_fra_agreement,
        "B_ACCELERATED":   b_accelerated,
        "LOTS_NUMBER":       lots_number,
        "YEAR":              year,
        "CRIT_PRICE_WEIGHT": crit_price_weight,
        "CRIT_CODE":         crit_code,
        "TOP_TYPE":          top_type,
        "ISO_COUNTRY_CODE":  iso_country_code,
        "TYPE_OF_CONTRACT":  type_of_contract,
        "CAE_TYPE":          cae_type,
        "MAIN_ACTIVITY":     main_activity,
    }

    with st.spinner("Appel à l'API..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            response.raise_for_status()             # lève une exception si status != 200
            result = response.json()                # parse le JSON retourné

            accepted   = result.get("accepted")
            confidence = result.get("confidence", 0)

            # ── Carte résultat principale ─────────────────────────────────────────────
            if accepted:
                st.success("✅ Tender accepté")
            else:
                st.error("❌ Tender rejeté")

            # ── Gauge de confiance ────────────────────────────────────────────────────
            st.markdown(f"### Confiance : {confidence:.0%}")
            st.progress(confidence)

            # ── Détail coloré selon le seuil ──────────────────────────────────────────
            if confidence >= 0.75:
                st.info("🟢 Signal fort — le modèle est très confiant.")
            elif confidence >= 0.55:
                st.warning("🟡 Signal modéré — à analyser avec précaution.")
            else:
                st.error("🔴 Signal faible — résultat incertain.")

            # ── Récapitulatif des inputs clés ─────────────────────────────────────────
            with st.expander("📋 Détail de la requête"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Pays",     result["input"]["ISO_COUNTRY_CODE"])
                    st.metric("Contrat",  result["input"]["TYPE_OF_CONTRACT"])
                    st.metric("Année",    int(result["input"]["YEAR"]))
                with col_b:
                    st.metric("Lots",     int(result["input"]["LOTS_NUMBER"]))
                    st.metric("Poids prix", f"{result['input']['CRIT_PRICE_WEIGHT']}%")
                    st.metric("Activité", result["input"]["MAIN_ACTIVITY"])                         # affiche la réponse complète

        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout — l'API ne répond pas.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Erreur HTTP {e.response.status_code} : {e.response.text}")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
