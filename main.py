from flask import Flask, request, jsonify
import pandas as pd
import joblib
import openai
import os
import re

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

thresholds = {
    "Heartrate": (60, 100),
    "Breathrate": (12, 20),
    "SPO2": (95, 100),
    "Walking_steps": (3000, None),
    "Caloricexpenditure": (100, None)
}

model = joblib.load("xgboost_emergency_model.pkl")

def detect_abnormal_features(data, thresholds):
    abnormal = {}
    for key, value in data.items():
        min_val, max_val = thresholds.get(key, (None, None))
        if min_val is not None and value < min_val:
            abnormal[key] = f"{value} ↓ (>= {min_val})"
        elif max_val is not None and value > max_val:
            abnormal[key] = f"{value} ↑ (<= {max_val})"
    return abnormal

def clean_output(text):
    text = re.sub(r"사회복지사.+?다[.]", "", text)
    text = re.sub(r"(\d+)\.\s*", r"\n\1. ", text)
    text = re.sub(r"\.+", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith("다.다."):
        text = text[:-3] + "다."
    elif text.endswith("다.") and len(text) > 2 and text[-3] == ".":
        text = text[:-1]
    return text

@app.route("/ai/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        required_features = ["Heartrate", "Breathrate", "SPO2", "Walking_steps", "Caloricexpenditure"]

        input_data = {}
        for feat in required_features:
            if feat not in data:
                return jsonify({"error": f"'{feat}' is required."}), 400
            try:
                input_data[feat] = float(data[feat])
            except ValueError:
                return jsonify({"error": f"'{feat}' must be a float."}), 400

        input_df = pd.DataFrame([input_data])
        prob = model.predict_proba(input_df)[0][1]
        prediction = int(prob > 0.7)
        label_map = {0: "Non-Emergency (정상)", 1: "Emergency (비정상)"}

        result = {
            "prediction_label": label_map[prediction],
            "probability": float(round(prob, 4)),
            "explanation": ""
        }

        if prediction == 1:
            abnormal_dict = detect_abnormal_features(input_data, thresholds)
            abnormal_text = "\n".join([f"{k}: {v}" for k, v in abnormal_dict.items()])

            importances = model.feature_importances_
            feature_names = model.feature_names_in_
            important_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            important_feature_text = ", ".join([feat for feat, _ in important_features[:2]])

            feature_text = "\n".join([f"{key}: {input_data[key]}" for key in feature_names])

            system_prompt = (
                "당신은 독거노인의 건강을 모니터링하는 시스템입니다. "
                "아래 생체 정보를 기반으로 사회복지사가 이해할 수 있도록 건강 상태를 간단히 설명하고, 필요한 생활 조치를 안내해주세요."
            )
            user_prompt = (
                f"독거노인의 건강 데이터:\n{feature_text}\n\n"
                f"이상 감지 항목:\n{abnormal_text}\n\n"
                f"특히 '{important_feature_text}' 항목이 중요합니다."
            )

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            result["explanation"] = clean_output(response.choices[0].message.content)
        else:
            result["explanation"] = "환자 상태는 정상입니다. 별도의 조치가 필요 없습니다."

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
