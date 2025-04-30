import joblib
import pandas as pd
import json
import openai
import os
import re

openai.api_key = os.getenv("OPENAI_API_KEY")  

thresholds = {
    "Heartrate": (60, 100),
    "Breathrate": (12, 20),
    "SPO2": (95, 100),
    "Skin_Temperature": (31, 33),
    "Sleep_phase": (1, 4),
    "Sleep_score": (80, 100),
    "Walking_steps": (3000, None),
    "Stress_index": (0, 25),
    "Activity_intensity": (200, None),
    "Caloricexpenditure": (100, None)
}

def detect_abnormal_features(data, thresholds):
    abnormal = {}
    for key, value in data.items():
        if key not in thresholds:
            continue
        min_val, max_val = thresholds[key]
        if min_val is not None and value < min_val:
            abnormal[key] = f"{value} ↓ (정상: ≥{min_val})"
        elif max_val is not None and value > max_val:
            abnormal[key] = f"{value} ↑ (정상: ≤{max_val})"
    return abnormal

def clean_output(text):
    text = re.sub(r"사회복지사.+?다[.]", "", text)
    text = re.sub(r"(\d+)\.\s*", r"\n\1. ", text)
    text = re.sub(r"\.+", ".", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if text.endswith("다.다."):
        text = text[:-3] + "다."
    elif text.endswith("다."):
        text = text[:-1] if text[-2] == "." else text
    return text.strip()

xgb_model = joblib.load("/home/dm-potato/dm-geon/final_capston/emergency_detection_caption/xgboost_emergency_model.pkl")

input_data = {
    "Heartrate": 75,
    "Breathrate": 16,
    "SPO2": 56,
    # "Skin_Temperature": 32,
    # "Sleep_phase": 2,
    "Sleep_score": 88,
    "Walking_steps": 5000,
    "Stress_index": 15,
    "Activity_intensity": 250,
    "Caloricexpenditure": 120
}
input_df = pd.DataFrame([input_data])
proba = xgb_model.predict_proba(input_df)[0][1]
print(f"Emergency Probability: {proba:.2f}")

prediction = int(proba > 0.7)

prediction = xgb_model.predict(input_df.fillna(0))[0]
label_map = {0: "Non-Emergency (정상)", 1: "Emergency (비정상)"}
output_result = {"prediction_label": label_map[prediction], "explanation": ""}

abnormal_dict = detect_abnormal_features(input_data, thresholds)
abnormal_text = "\n".join([f"{k}: {v}" for k, v in abnormal_dict.items()])

if prediction == 1:
    importances = xgb_model.feature_importances_
    feature_names = xgb_model.feature_names_in_
    important_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    important_feature_text = ", ".join([feat for feat, _ in important_features[:2]])

    feature_text = "\n".join([f"{key}: {input_data.get(key, 'N/A')}" for key in feature_names])
    system_prompt = (
        "당신은 독거노인의 건강을 모니터링하는 시스템입니다. "
        "아래 정보를 바탕으로 사회복지사가 쉽게 이해할 수 있도록 건강 상태를 간단하게 설명하고, "
        "생활습관이나 간단한 조치 방법을 안내해주세요."
    )
    user_prompt = (
        "아래는 한 독거노인의 생체 신호 데이터입니다:\n"
        f"{feature_text}\n\n"
        f"이 중에서 다음 항목에 이상이 감지되었습니다:\n{abnormal_text}\n\n"
        f"특히 '{important_feature_text}' 항목의 중요도가 높습니다."
    )

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9
    )

    explanation = response.choices[0].message.content.strip()
    output_result["explanation"] = clean_output(explanation)

else:
    output_result["explanation"] = "환자 상태는 정상입니다. 별도의 조치가 필요 없습니다."

print("\n최종 결과 (JSON):")
print(json.dumps(output_result, ensure_ascii=False, indent=4))