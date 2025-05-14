from flask import Flask, request, jsonify
import pandas as pd
import joblib
import openai
import os
import re
import requests
from datetime import datetime

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

thresholds = {
    "Heartrate": (60, 100),
    "Breathrate": (12, 20),
    "SPO2": (95, 100),
    "Walking_steps": (3000, None),
    "Caloricexpenditure": (100, None)
}

model = joblib.load("/app/model/xgboost_emergency_model.pkl")

def detect_abnormal_features(data, thresholds, timestamp_str):
    abnormal = {}
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
    except Exception:
        timestamp = None

    for key, value in data.items():
        if key in ["Walking_steps", "Caloricexpenditure"]:
            if not (timestamp and timestamp.hour == 23 and timestamp.minute == 59):
                continue

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
        elderly_id = data.get("elderlyId")
        timestamp_str = data.get("timestamp")

        if elderly_id is None:
            return jsonify({"error": "'elderlyId' is required."}), 400
        if timestamp_str is None:
            return jsonify({"error": "'timestamp' (e.g., '2025-05-14 23:59') is required."}), 400

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
        prediction_label = label_map[prediction]
        explanation_text = ""

        if prediction == 1:
            abnormal_dict = detect_abnormal_features(input_data, thresholds, timestamp_str)
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
            explanation_text = clean_output(response.choices[0].message.content)
        else:
            explanation_text = "환자 상태는 정상입니다. 별도의 조치가 필요 없습니다."

        payload = {
            "elderlyId": elderly_id,
            "predictionLabel": prediction_label,
            "explanation": explanation_text
        }

        try:
            backend_url = "https://server.lifewatch.store/api/alert/emergency"
            headers = {"Content-Type": "application/json"}
            backend_response = requests.post(backend_url, json=payload, headers=headers)
            if backend_response.status_code != 200:
                print("⚠️ 백엔드 응답 실패:", backend_response.status_code, backend_response.text)
            else:
                print("✅ 백엔드 전송 성공:", backend_response.status_code)
        except Exception as be:
            print("⚠️ 백엔드 전송 예외:", str(be))

        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
