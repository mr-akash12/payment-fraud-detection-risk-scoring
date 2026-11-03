import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Fraud Detection", layout="wide")

# --------------------------------------------------
# GLOBAL BACKGROUND COLOR
# --------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0f2d44;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #081627 !important;
}
h1, h2, h3, p, label, span, div {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LEFT PANEL (SIDEBAR)
# --------------------------------------------------
with st.sidebar:
    st.header("📂 Upload File")
    st.write(
        "Upload a CSV file with transaction data to get fraud predictions, "
        "probabilities, and risk scores."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# --------------------------------------------------
# MAIN CONTENT HEADER
# --------------------------------------------------
st.title("💳 Fraud Detection and Risk Scoring Dashboard")

st.image(
    "sumup-n2qYTKRTBys-unsplash.jpg",
    use_container_width=True
)

# --------------------------------------------------
# PROCESS FILE WHEN UPLOADED
# --------------------------------------------------
if uploaded_file:

    # Load CSV
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Load model
    @st.cache_resource
    def load_model():
        return joblib.load("fraud_detection_model (1).pkl")

    pipeline = load_model()
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    # Preprocess input
    X = data.drop(columns=["isFraud", "isfraud"], errors="ignore")

    # -----------------------------
    # Fill missing columns expected by preprocessor
    # -----------------------------
    if "num" in preprocessor.named_transformers_:
        num_cols = preprocessor.named_transformers_["num"].feature_names_in_
        for col in num_cols:
            if col not in X.columns:
                X[col] = 0
            else:
                X[col] = X[col].fillna(0)

    if "cat" in preprocessor.named_transformers_:
        cat_cols = preprocessor.named_transformers_["cat"].feature_names_in_
        for col in cat_cols:
            if col not in X.columns:
                X[col] = "missing"
            else:
                X[col] = X[col].fillna("missing")

    # Transform
    Xp = preprocessor.transform(X)

    # -----------------------------
    # Predictions before threshold
    # -----------------------------
    preds_prob = model.predict_proba(Xp)[:, 1]

    # --------------------------------------------------
    # Compute Optimal Threshold using Youden's J
    # --------------------------------------------------
    label_col = None
    for col in ["isFraud", "isfraud"]:
        if col in data.columns:
            label_col = col
            break

    if label_col is not None:
        y_true_all = data[label_col]
        fpr_all, tpr_all, roc_thresholds_all = roc_curve(y_true_all, preds_prob)
        J = tpr_all - fpr_all
        ix = np.argmax(J)
        best_threshold = roc_thresholds_all[ix]
    else:
        best_threshold = 0.5

    # --------------------------------------------------
    # SIDEBAR THRESHOLD SLIDER
    # --------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        st.header("⚖️ Prediction Threshold")
        threshold = st.slider(
            "Set fraud probability threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(best_threshold),
            step=0.01,
            help=f"Optimal threshold from ROC = {best_threshold:.3f}"
        )

    # -----------------------------
    # Final predictions using chosen threshold
    # -----------------------------
    preds = (preds_prob >= threshold).astype(int)

    # Dynamic risk levels
    high_risk_cutoff = max(threshold + 0.3, 0.8)
    risk = np.where(preds_prob >= high_risk_cutoff, "High",
             np.where(preds_prob >= threshold, "Medium", "Low"))

    # Store results
    results = data.copy()
    results["fraud_probability"] = preds_prob
    results["fraud_prediction"] = preds
    results["risk_level"] = risk

    # --------------------------------------------------
    # TABS LAYOUT
    # --------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["Overview", "Predictions", "Visualizations"])

    # ---------- TAB 1 ----------
    with tab1:
        st.subheader("📊 Overview")
        st.write("Total Transactions:", len(results))
        st.write("Predicted Fraud Cases:", results["fraud_prediction"].sum())
        st.write("Risk Level Distribution:")
        st.dataframe(results["risk_level"].value_counts())

    # ---------- TAB 2 ----------
    with tab2:
        st.subheader("🔍 Predictions Summary")

        def color_risk(val):
            if val == "High":
                return 'color: red; font-weight: bold'
            elif val == "Medium":
                return 'color: #c7b700; font-weight: bold'
            else:
                return 'color: #4da6ff; font-weight: bold'

        prediction_results = results[["fraud_prediction", "fraud_probability", "risk_level"]]
        st.dataframe(prediction_results.style.applymap(color_risk, subset=["risk_level"]))

        csv = prediction_results.to_csv(index=False).encode()
        st.download_button("⬇️ Download Predictions CSV", csv, "predictions_results.csv", "text/csv")

    # ---------- TAB 3 ----------
    with tab3:
        st.subheader("🏆 Top 10 Feature Importances")

        try:
            importances = model.feature_importances_
            transformed_names = []

            if "num" in preprocessor.named_transformers_:
                num_trans = preprocessor.named_transformers_["num"]
                if hasattr(num_trans, "get_feature_names_out"):
                    transformed_names.extend(num_trans.get_feature_names_out())
                else:
                    transformed_names.extend(X.columns)

            if "cat" in preprocessor.named_transformers_:
                ohe = preprocessor.named_transformers_["cat"]
                transformed_names.extend(ohe.get_feature_names_out())

            fi = pd.Series(importances, index=transformed_names)
            fi_top10 = fi.sort_values(ascending=False).head(10)

            import matplotlib.cm as cm
            cmap = cm.get_cmap("viridis", len(fi_top10))
            colors = [cmap(i) for i in range(len(fi_top10))]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=fi_top10.values, y=fi_top10.index, ax=ax, palette=colors)
            ax.set_title("Top 10 Most Important Features")
            st.pyplot(fig)

        except Exception as e:
            st.error("Feature importances unavailable.")
            st.write(e)

        # Fraud count chart
        st.subheader("📦 Fraud Prediction Counts")
        fraud_counts = results["fraud_prediction"].value_counts().rename(index={0: "Not Fraud", 1: "Fraud"})
        fraud_colors = {"Fraud": "red", "Not Fraud": "#4da6ff"}
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x=fraud_counts.index,
            y=fraud_counts.values,
            palette=[fraud_colors[label] for label in fraud_counts.index],
            ax=ax1
        )
        st.pyplot(fig1)

        # Risk level chart
        st.subheader("🎯 Risk Level Distribution")
        risk_counts = results["risk_level"].value_counts()
        risk_colors = {"High": "red", "Medium": "#c7b700", "Low": "#4da6ff"}
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x=risk_counts.index,
            y=risk_counts.values,
            palette=[risk_colors[r] for r in risk_counts.index],
            ax=ax2
        )
        st.pyplot(fig2)

        # Confusion matrix
        st.subheader("🧮 Confusion Matrix")
        if label_col is None:
            st.warning("No true fraud labels found in your data.")
        else:
            y_true = data[label_col]
            y_pred = results["fraud_prediction"]

            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=["Not Fraud", "Fraud"])
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax3, cmap="Blues", colorbar=False)
            st.pyplot(fig3)

        # --------------------------------------------------
        # 🚨 Fraudulent Transactions (Replacing ROC Curve)
        # --------------------------------------------------
        st.subheader("🚨 Transactions Predicted as Fraud")

        fraud_cases = results[results["fraud_prediction"] == 1]

        if fraud_cases.empty:
            st.success("No fraudulent transactions detected at this threshold.")
        else:
            st.write(f"Total Fraudulent Transactions: {len(fraud_cases)}")
            st.dataframe(fraud_cases.head(50))

            csv_fraud = fraud_cases.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Fraudulent Transactions",
                csv_fraud,
                "fraudulent_transactions.csv",
                "text/csv"
            )
