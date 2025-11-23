from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# -------------------------
# Load saved model
# -------------------------
pipeline = joblib.load("credit_risk_model.pkl")  # make sure this file exists

# -------------------------
# Helper functions
# -------------------------
def get_risk_level(pd_score):
    if pd_score < 0.2:
        return "low"
    elif pd_score < 0.5:
        return "medium"
    else:
        return "high"

def get_credit_grade(pd_score):
    if pd_score < 0.10:
        return "A"
    elif pd_score < 0.20:
        return "B"
    elif pd_score < 0.35:
        return "C"
    elif pd_score < 0.50:
        return "D"
    elif pd_score < 0.70:
        return "E"
    else:
        return "F"

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return "Credit Risk API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df_input = pd.DataFrame([data])

        # ---- Rename columns to match pipeline ----
        df_input.rename(columns={
            "Credit_amount": "Credit amount",
            "Checking_account": "Checking account",
            "Saving_accounts": "Saving accounts"
        }, inplace=True)

        # ---- Feature engineering ----
        df_input["Age_bucket"] = pd.cut(df_input["Age"], bins=[18,25,35,45,55,65,100], labels=False)
        df_input["Credit_per_month"] = df_input["Credit amount"] / df_input["Duration"]

        # ---- Prediction ----
        pd_score = float(pipeline.predict_proba(df_input)[0][1])

        return jsonify({
            "default_probability": round(pd_score, 4),
            "risk_level": get_risk_level(pd_score),
            "credit_grade": get_credit_grade(pd_score)
        })

    except Exception as e:
        # Return error info as JSON (for debugging)
        return jsonify({"error": str(e)}), 500

# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
