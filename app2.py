
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="ModuFly AI",
    page_icon="🪰",
    layout="wide"
)

# ---------- 1. PROJECT HEADER ----------

st.title("🪰 ModuFly AI")
st.subheader(
    "AI-powered modular Black Soldier Fly waste-to-value system"
)

st.info(
    "Hackathon prototype: readings and predictions are simulated. "
    "This demo is not a validated farm-control or safety system."
)

# ---------- 2. SIMULATED FARM MODULES ----------

rng = np.random.default_rng(42)

modules = pd.DataFrame({
    "Module": [f"Module {i}" for i in range(1, 7)],
    "Temperature (C)": rng.normal(29, 3, 6).round(1),
    "Humidity (%)": rng.normal(65, 10, 6).clip(20, 100).round(1),
    "Waste Input (kg)": rng.uniform(10, 50, 6).round(1),
})

# Add one unusual reading so the demo is easy to explain.
modules.loc[5, "Temperature (C)"] = 40.0

# ---------- 3. SIDEBAR INPUTS ----------

st.sidebar.header("Farm settings")

waste_kg = st.sidebar.number_input(
    "Organic waste input (kg)",
    min_value=0.0,
    max_value=10000.0,
    value=100.0,
    step=10.0
)

yield_percent = st.sidebar.slider(
    "Assumed wet larval biomass yield (%)",
    min_value=1,
    max_value=40,
    value=15
)

product_price = st.sidebar.number_input(
    "Illustrative biomass price (₹/kg)",
    min_value=0.0,
    max_value=100000.0,
    value=50.0,
    step=5.0
)

daily_cost = st.sidebar.number_input(
    "Illustrative operating cost (₹)",
    min_value=0.0,
    max_value=1000000.0,
    value=500.0,
    step=50.0
)

# ---------- 4. KEY PERFORMANCE INDICATORS ----------

estimated_biomass = waste_kg * yield_percent / 100
estimated_revenue = estimated_biomass * product_price
estimated_margin = estimated_revenue - daily_cost

c1, c2, c3, c4 = st.columns(4)

c1.metric("Modules monitored", len(modules))
c2.metric("Waste input", f"{waste_kg:,.1f} kg")
c3.metric("Estimated wet biomass", f"{estimated_biomass:,.1f} kg")
c4.metric("Illustrative margin", f"₹{estimated_margin:,.0f}")

st.caption(
    "Biomass and margin are scenario estimates, not measured "
    "production or guaranteed revenue."
)

# ---------- 5. MODULE MONITOR ----------

st.header("1. Modular Farm Monitor")

st.write("Example environmental readings for six simulated modules.")

st.dataframe(modules, use_container_width=True)

st.bar_chart(
    modules.set_index("Module")[["Temperature (C)"]]
)

# ---------- 6. RULE-BASED ALERTS ----------

st.header("2. Monitoring Alerts")

temperature_limit = st.number_input(
    "Demo temperature alert threshold (°C)",
    min_value=20.0,
    max_value=50.0,
    value=35.0,
    step=1.0
)

humidity_limit = st.number_input(
    "Demo high-humidity alert threshold (%)",
    min_value=40.0,
    max_value=100.0,
    value=85.0,
    step=5.0
)

alerts = []

for _, row in modules.iterrows():
    if row["Temperature (C)"] > temperature_limit:
        alerts.append({
            "Module": row["Module"],
            "Alert": "High temperature",
            "Reading": row["Temperature (C)"]
        })

    if row["Humidity (%)"] > humidity_limit:
        alerts.append({
            "Module": row["Module"],
            "Alert": "High humidity",
            "Reading": row["Humidity (%)"]
        })

if alerts:
    st.warning(f"{len(alerts)} demo alert(s) detected.")
    st.dataframe(pd.DataFrame(alerts), use_container_width=True)
else:
    st.success("No alerts triggered by the configured demo thresholds.")

st.caption(
    "Thresholds are configurable demonstration settings, "
    "not validated biological safety limits."
)

# ---------- 7. AI ANOMALY DETECTION ----------

st.header("3. AI Anomaly Detection")

st.write(
    "Isolation Forest is an unsupervised machine-learning algorithm "
    "that identifies readings that differ from the reference data."
)

features = ["Temperature (C)", "Humidity (%)"]

# Synthetic reference observations for demonstrating the ML pipeline.
reference = pd.DataFrame({
    "Temperature (C)": rng.normal(29, 2.5, 300),
    "Humidity (%)": rng.normal(65, 8, 300).clip(20, 100)
})

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(reference[features])

modules["AI Result"] = model.predict(modules[features])
modules["AI Anomaly"] = modules["AI Result"].map({
    1: "Typical",
    -1: "Unusual"
})

st.dataframe(
    modules[
        ["Module", "Temperature (C)", "Humidity (%)", "AI Anomaly"]
    ],
    use_container_width=True
)

unusual_count = (modules["AI Result"] == -1).sum()

st.metric("Modules flagged as unusual by AI", int(unusual_count))

st.caption(
    "The model is trained on synthetic reference data. Its flags "
    "demonstrate the software workflow and do not prove a real farm problem."
)

# ---------- 8. WASTE-TO-VALUE CALCULATOR ----------

st.header("4. Waste-to-Value Calculator")

st.write(f"Organic waste entered: **{waste_kg:,.1f} kg**")
st.write(
    f"Assumed wet biomass yield: **{yield_percent}%**"
)
st.write(
    f"Estimated wet larval biomass: **{estimated_biomass:,.1f} kg**"
)
st.write(
    f"Illustrative biomass revenue: **₹{estimated_revenue:,.2f}**"
)
st.write(
    f"Illustrative margin after entered costs: **₹{estimated_margin:,.2f}**"
)

st.caption(
    "This simplified calculator excludes many real-world costs, "
    "moisture changes, processing losses, and regulatory requirements. "
    "Wet biomass is not the same as dried feed meal."
)

# ---------- 9. BATCH RECORDS AND EXPORT ----------

st.header("5. Batch Records")

batch_records = modules.copy()
batch_records["Batch ID"] = [
    f"MF-2026-{i:03d}" for i in range(1, len(modules) + 1)
]
batch_records["Data Source"] = "Simulated"
batch_records["Waste Input (kg)"] = modules["Waste Input (kg)"]

st.dataframe(batch_records, use_container_width=True)

csv_data = batch_records.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download batch records as CSV",
    data=csv_data,
    file_name="modufly_batch_records.csv",
    mime="text/csv"
)

# ---------- 10. FOOTER ----------

st.divider()
st.write("**ModuFly AI | Hackathon MVP**")
st.write(
    "Future scope: real sensors, validated production data, "
    "batch traceability, and improved yield prediction."
)
