from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

app = Flask(__name__)

# تحميل النموذج وال scaler
with open('model/copper_model.pkl', 'rb') as model_file:
    model, scaler = pickle.load(model_file)

# ملف البيانات
DATA_FILE = Path(__file__).resolve().parent / 'data' / 'copper_prediction_dataset_1000.csv'

# أعمدة النموذج
feature_columns = [
    'global_demand_index',
    'oil_price',
    'usd_index',
    'china_industry_output',
    'energy_cost_index',
    'market_sentiment',
    'supply_disruption_index'
]

# مفتاح OilPriceAPI
OIL_API_KEY = "980ecfe16b13b1881b03de30115dbb59897c0da5c5333e30717343e244ad7927"


# ---------------------------------------------------------
# دالة تنظيف البيانات
# ---------------------------------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # إزالة الصفوف المكررة
    df = df.drop_duplicates()
    # إزالة الصفوف التي تحتوي قيم مفقودة
    df = df.dropna()
    # إعادة ضبط الفهارس
    df = df.reset_index(drop=True)
    return df



#  جلب سعر النفط Brent من OilPriceAPI

def get_brent_price():
    try:
        url = "https://api.oilpriceapi.com/v1/prices/latest"
        headers = {"Authorization": f"Token {OIL_API_KEY}"}
        response = requests.get(url, headers=headers)
        data = response.json()
        return float(data["data"]["price"])
    except Exception as e:
        print("Error fetching Brent:", e)
        return 75.0  # fallback


# ---------------------------------------------------------
# 2) جلب مؤشر الدولار DXY من Yahoo Finance (بدون مفتاح)
# ---------------------------------------------------------
def get_dxy_value():
    try:
        ticker = yf.Ticker("DX-Y.NYB")
        data = ticker.history(period="1d")
        if data.empty:
            return 102.0  # fallback
        return float(data["Close"].iloc[-1])
    except Exception as e:
        print("Error fetching DXY:", e)
        return 102.0  # fallback


# ---------------------------------------------------------
# صفحة Dashboard
# ---------------------------------------------------------
@app.route('/dashboard')
def dashboard():
    return render_template('index.html')


@app.route('/dashboard-data')
def dashboard_data():
    df = pd.read_csv(DATA_FILE)
    df = clean_dataframe(df)  # تنظيف البيانات

    # آخر 60 يوم
    if 'date' in df.columns:
        history_dates = df['date'].tail(60).tolist()
    else:
        history_dates = list(range(len(df.tail(60))))

    history_prices = df['copper_price'].tail(60).tolist()

    oil_price = get_brent_price()
    usd_index = get_dxy_value()

    try:
        news_sentiment = get_market_sentiment_from_news()
        sentiment = float(news_sentiment) if news_sentiment is not None else (oil_price / 100) - (usd_index / 200)
    except Exception as e:
        print("Error getting news sentiment in dashboard_data:", e)
        sentiment = (oil_price / 100) - (usd_index / 200)

    last_row = df[feature_columns].tail(1).values
    last_scaled = scaler.transform(last_row)
    predicted_price = model.predict(last_scaled)[0]

    return jsonify({
        "predicted_price": predicted_price,
        "oil_price": oil_price,
        "usd_index": usd_index,
        "market_sentiment": sentiment,
        "history_dates": history_dates,
        "history_prices": history_prices
    })


# ---------------------------------------------------------
# صفحة التوقع اليدوي
# ---------------------------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    data = request.get_json(force=True)

    features = [
        float(data['global_demand_index']),
        float(data['oil_price']),
        float(data['usd_index']),
        float(data['china_industry_output']),
        float(data['energy_cost_index']),
        float(data['market_sentiment']),
        float(data['supply_disruption_index'])
    ]

    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]
    coefficients = model.coef_.tolist()

    return jsonify({
        "predicted_copper_price": prediction,
        "coefficients": coefficients
    })


@app.route('/manual')
def manual():
    return render_template('predict.html')


@app.route("/auto")
def auto_page():
    return render_template("api_predict.html")


@app.route('/model_info')
def model_info_page():
    return render_template('model_info.html')


@app.route('/model-info-data')
def model_info_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df = clean_dataframe(df)  # تنظيف البيانات

        if not set(feature_columns).issubset(df.columns) or 'copper_price' not in df.columns:
            return jsonify({'success': False, 'error': 'Required columns missing in dataset'}), 400

        X = df[feature_columns].values
        y = df['copper_price'].values

        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)

        mse = mean_squared_error(y, preds)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y, preds)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y - preds) / np.where(y == 0, np.nan, y))) * 100
            if np.isnan(mape):
                mape = None

        summary = {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'target_mean': float(np.nanmean(y)),
            'target_std': float(np.nanstd(y))
        }

        coefficients = getattr(model, 'coef_', None)
        intercept = getattr(model, 'intercept_', None)

        return jsonify({
            'success': True,
            'metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': None if mape is None else float(mape)
            },
            'summary': summary,
            'coefficients': coefficients.tolist() if coefficients is not None else None,
            'intercept': float(intercept) if intercept is not None else None,
            'feature_columns': feature_columns
        })
    except Exception as e:
        print('Error in model_info_data:', e)
        return jsonify({'success': False, 'error': str(e)}), 500


def get_market_sentiment_from_news():
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "commodities OR copper OR metals OR economy",
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": "d13a4e5c597c4073ac9906f7bf274901"
        }
        response = requests.get(url, params=params)
        data = response.json()

        articles = []
        for a in data.get("articles", []):
            title = a.get("title") or ""
            desc = a.get("description") or ""
            combined = (title + " " + desc).strip()
            if combined:
                articles.append(combined)

        positive_words = ["growth", "increase", "strong", "positive", "recovery"]
        negative_words = ["decline", "drop", "weak", "negative", "crisis"]

        score = 0
        for text in articles:
            text_lower = text.lower()
            score += sum(1 for w in positive_words if w in text_lower)
            score -= sum(1 for w in negative_words if w in text_lower)

        return score / 10
    except Exception as e:
        print("NewsAPI error:", e)
        return 0.0


@app.route("/api_predict")
def auto_predict():
    try:
        df = pd.read_csv(DATA_FILE)
        df = clean_dataframe(df)  # تنظيف البيانات

        oil_price = get_brent_price()
        usd_index = get_dxy_value()
        news_sentiment = get_market_sentiment_from_news()

        other = df[[col for col in feature_columns if col not in ["oil_price", "usd_index", "market_sentiment"]]].mean().values.tolist()

        full_features = [
            other[0],
            oil_price,
            usd_index,
            other[1],
            other[2],
            news_sentiment,
            other[3]
        ]

        scaled = scaler.transform([full_features])
        prediction = model.predict(scaled)[0]
        features_dict = {
            'global_demand_index': full_features[0],
            'oil_price': full_features[1],
            'usd_index': full_features[2],
            'china_industry_output': full_features[3],
            'energy_cost_index': full_features[4],
            'market_sentiment': full_features[5],
            'supply_disruption_index': full_features[6]
        }

        # Return both top-level keys and a `features_used` object for the frontend
        return jsonify({
            'predicted_copper_price': float(prediction),
            'oil_price': features_dict['oil_price'],
            'usd_index': features_dict['usd_index'],
            'market_sentiment': features_dict['market_sentiment'],
            'features_used': features_dict
        })
    except Exception as e:
        print('Error in auto_predict:', e)
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------
# Upload CSV and merge with existing dataset
# ---------------------------------------------------------
@app.route('/upload', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'GET':
        return render_template('upload.html')

    # POST: handle uploaded CSV
    try:
        file = request.files.get('csv_file')
        if not file:
            return render_template('upload.html', flash_message='لم يتم إرسال ملف', flash_success=False)

        # read uploaded CSV into DataFrame
        new_df = pd.read_csv(file)

        # read existing data and concat
        existing_df = pd.read_csv(DATA_FILE)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = clean_dataframe(combined)

        # save back
        combined.to_csv(DATA_FILE, index=False)

        stats = {
            'rows': int(combined.shape[0]),
            'columns': int(combined.shape[1]),
            'headers': combined.columns.tolist(),
            'columns_match': set(feature_columns).issubset(combined.columns)
        }

        return render_template('upload.html', flash_message='تم رفع ودمج الملف بنجاح', flash_success=True, stats=stats)
    except Exception as e:
        print('Error in upload_csv:', e)
        return render_template('upload.html', flash_message=f'حدث خطأ أثناء الرفع: {e}', flash_success=False)


# ---------------------------------------------------------
# إعادة تدريب النموذج (تشغيل السكربت)
# ---------------------------------------------------------
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        import subprocess, sys
        script_path = Path(__file__).resolve().parent / 'model' / 'train_model.py'
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, timeout=600)
        out = result.stdout
        err = result.stderr
        if result.returncode != 0:
            print('Retrain stderr:', err)
            return render_template('upload.html', flash_message=f'فشل إعادة التدريب: {err}', flash_success=False)
        # successful
        return render_template('upload.html', flash_message='اكتملت إعادة تدريب النموذج بنجاح', flash_success=True)
    except Exception as e:
        print('Error in retrain:', e)
        return render_template('upload.html', flash_message=f'خطأ أثناء إعادة التدريب: {e}', flash_success=False)



if __name__ == '__main__':
    # تشغيل التطبيق محليًا عند استدعاء الملف مباشرة
    # ضبط debug=True مفيد أثناء التطوير، يمكنك تغييره إلى False في الإنتاج
    app.run(host='0.0.0.0', port=5000, debug=True)

    

