import streamlit as st
import requests
import random
import streamlit.components.v1 as components



st.set_page_config(page_title="Vectyfi · Tender Prediction", layout="centered")

API_URL = "https://vectyfi-api-828368828432.europe-west1.run.app"

TOP_TYPES       = ["OPE", "AWP", "NIC", "RES", "NOP", "NOC", "NIP", "COD", "INP"]
COUNTRIES        = ["FR", "DE", "IT", "ES", "BE", "NL", "PL", "PT", "SE", "AT"]
CONTRACT_TYPES  = ["U", "S", "W"]
CAE_TYPES        = ["OTH", "RA", "LA", "PUB", "DEF", "CGA", "HEA", "EU", "REG"]
MAIN_ACTIVITIES = [
    "Health", "Defence", "Railway services", "Other",
    "General public\\services", "Education", "Environment",
    "Urban railway, tramway, trolleybus or bus services",
    "Housing and community amenities", "Recreation, culture and religion"
]

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
st.caption("Fill in the fields or generate random data, then call the API.")

# ── Bouton Random ─────────────────────────────────────────────────────────────
if st.button("🎲 Generate Random Data"):
    st.session_state.inputs = random_input()

st.divider()

# ── Formulaire ────────────────────────────────────────────────────────────────
inp = st.session_state.inputs

st.subheader("Binary fields")
col1, col2 = st.columns(2)
with col1:
    b_multiple_cae  = st.selectbox("Multiple CAE",  ["Y", "N"], index=["Y","N"].index(inp["B_MULTIPLE_CAE"]))
    b_eu_funds      = st.selectbox("EU Funds",      ["Y", "N"], index=["Y","N"].index(inp["B_EU_FUNDS"]))
    b_gpa           = st.selectbox("GPA",           ["Y", "N"], index=["Y","N"].index(inp["B_GPA"]))
with col2:
    b_fra_agreement = st.selectbox("FRA Agreement", ["Y", "N"], index=["Y","N"].index(inp["B_FRA_AGREEMENT"]))
    b_accelerated   = st.selectbox("Accelerated",   ["Y", "N"], index=["Y","N"].index(inp["B_ACCELERATED"]))

st.subheader("Numerical fields")
col3, col4 = st.columns(2)
with col3:
    lots_number       = st.number_input("Lots Number",       value=inp["LOTS_NUMBER"],       min_value=1.0)
    year              = st.number_input("Year",              value=int(inp["YEAR"]),         min_value=2000, max_value=2030, step=1)
with col4:
    crit_price_weight = st.number_input("Crit Price Weight", value=inp["CRIT_PRICE_WEIGHT"], min_value=0.0, max_value=100.0)
    crit_code         = st.number_input("Crit Code",         value=inp["CRIT_CODE"],         min_value=1.0)

st.subheader("Categorical fields")
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
if st.button("🚀 Predict", type="primary"):
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

    with st.spinner("Calling API..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
            response.raise_for_status()             # lève une exception si status != 200
            result = response.json()                # parse le JSON retourné

            accepted   = result.get("accepted")
            confidence = result.get("confidence", 0)

            # ── Carte résultat principale ─────────────────────────────────────────────
            if accepted:
                st.success("✅ Tender accepted")
            else:
                st.error("❌ Tender rejected")

            # ── Gauge de confiance ────────────────────────────────────────────────────
            st.markdown(f"### Confidence : {confidence:.0%}")
            st.progress(confidence)

            # ── Détail coloré selon le seuil ──────────────────────────────────────────
            if confidence >= 0.75:
                st.info("🟢 Strong signal")
            elif confidence >= 0.55:
                st.warning("🟡 Moderate signal ")
            else:
                st.error("🔴 Weak signal")

            # ── Récapitulatif des inputs clés ─────────────────────────────────────────
            with st.expander("📋 Request details"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Pays",     result["input"]["ISO_COUNTRY_CODE"])
                    st.metric("Contrat",  result["input"]["TYPE_OF_CONTRACT"])
                    st.metric("Année",    int(result["input"]["YEAR"]))
                with col_b:
                    st.metric("Lots",     int(result["input"]["LOTS_NUMBER"]))
                    st.metric("Poids prix", f"{result['input']['CRIT_PRICE_WEIGHT']}%")
                    st.metric("Activité", result["input"]["MAIN_ACTIVITY"])

            with st.expander("🔍 SHAP Explanation"):
                force_html = result.get("force_plot_html")
                if force_html:
                    # Wrap in white background so SHAP plot is readable on dark theme
                    st.markdown("""
                    **How to read this chart:**
                    - The **base value** is the model's average prediction across all training data
                    - 🔴 **Red arrows** push the prediction **higher** (towards accepted)
                    - 🔵 **Blue arrows** push the prediction **lower** (towards rejected)
                    - The longer the arrow, the stronger the feature's impact
                    - The final score (right side) is the model's output for this tender
                    """)
                    st.divider()
                    wrapped = f"""
                    <div style="background-color: white; padding: 20px; border-radius: 8px;">
                        {force_html}
                    </div>
                    """
                    components.html(wrapped, height=350, scrolling=True)
                else:
                    st.warning("No SHAP explanation available.")

        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout — the API is not responding.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error {e.response.status_code} : {e.response.text}")
        except Exception as e:
            st.error(f"❌ Error : {e}")
