from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import sqlite3
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)

# تحميل النموذج وال scaler
with open('model/copper_model.pkl', 'rb') as model_file:
    model, scaler = pickle.load(model_file)

# قاعدة البيانات SQLite
DB_FILE = Path(__file__).resolve().parent / 'data' / 'copper_data.db'

# أعمدة النموذج - الآن تتضمن copper_price
feature_columns = [
    'global_demand_index',
    'oil_price',
    'usd_index',
    'china_industry_output',
    'energy_cost_index',
    'market_sentiment',
    'supply_disruption_index',
    'copper_price'
]

# مفتاح OilPriceAPI
OIL_API_KEY = "980ecfe16b13b1881b03de30115dbb59897c0da5c5333e30717343e244ad7927"


# ---------------------------------------------------------
# دالة لإعداد جداول قاعدة البيانات
# ---------------------------------------------------------
def init_predictions_table():
    """إنشاء جدول predictions إذا لم يكن موجوداً"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            global_demand_index REAL,
            oil_price REAL,
            usd_index REAL,
            china_industry_output REAL,
            energy_cost_index REAL,
            market_sentiment REAL,
            supply_disruption_index REAL,
            copper_price REAL,
            predicted_price REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# دالة لقراءة البيانات من SQLite
# ---------------------------------------------------------
def load_data_from_db():
    """قراءة البيانات من قاعدة بيانات SQLite"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM copper_data", conn)
    conn.close()
    return df


# ---------------------------------------------------------
# دالة لإعادة ترتيب جميع البيانات في قاعدة البيانات
# ---------------------------------------------------------
def reorder_copper_data():
    """إعادة ترتيب جميع البيانات في جدول copper_data حسب التاريخ"""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # قراءة جميع البيانات
        df = pd.read_sql_query("SELECT * FROM copper_data", conn)
        
        if df.empty or 'date' not in df.columns:
            conn.close()
            return
        
        # إنشاء عمود مؤقت للترتيب
        # محاولة تحويل التاريخ بطرق متعددة للتأكد من التحويل الصحيح
        # استخدام infer_datetime_format للتعرف التلقائي على التنسيق
        df['date_datetime'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        
        # فصل البيانات الصالحة عن غير الصالحة
        df_valid = df[df['date_datetime'].notna()].copy()
        df_invalid = df[df['date_datetime'].isna()].copy()
        
        # ترتيب البيانات الصالحة حسب التاريخ (تنازلي - من الأحدث إلى الأقدم)
        if not df_valid.empty:
            df_valid = df_valid.sort_values('date_datetime', ascending=False, na_position='last', kind='mergesort')
            df_valid = df_valid.reset_index(drop=True)
            # تحويل التاريخ إلى string بعد الترتيب
            df_valid['date'] = df_valid['date_datetime'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
            
            # طباعة أول وآخر تاريخ للتحقق (للتطوير فقط)
            if len(df_valid) > 0:
                print(f"[reorder_copper_data] أول تاريخ بعد الترتيب: {df_valid['date'].iloc[0]}, آخر تاريخ: {df_valid['date'].iloc[-1]}, عدد الصفوف: {len(df_valid)}")
        
        # دمج البيانات الصالحة وغير الصالحة
        # البيانات الصالحة أولاً (مرتبة تنازلياً - من الأحدث إلى الأقدم)، ثم البيانات غير الصالحة
        if not df_invalid.empty:
            df_invalid = df_invalid.reset_index(drop=True)
            # عند الترتيب التنازلي في قاعدة البيانات، البيانات غير الصالحة في النهاية
            df = pd.concat([df_valid, df_invalid], ignore_index=True)
        else:
            df = df_valid
        
        # إزالة العمود المؤقت
        if 'date_datetime' in df.columns:
            df = df.drop(columns=['date_datetime'])
        
        # تنظيف البيانات قبل الحفظ
        if 'date' in df.columns:
            df = df.drop_duplicates(subset=['date'], keep='last')
        if 'date' in df.columns:
            df = df[df['date'].notna()]
        df = df.reset_index(drop=True)
        
        # حفظ البيانات المرتبة مرة أخرى في قاعدة البيانات
        # التأكد من أن البيانات مرتبة بشكل صحيح قبل الحفظ
        if 'date' in df.columns and not df.empty:
            # التحقق من الترتيب مرة أخرى قبل الحفظ
            df['date_datetime_check'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
            df_sorted = df.sort_values('date_datetime_check', ascending=False, na_position='last', kind='mergesort')
            df_sorted = df_sorted.reset_index(drop=True)
            df_sorted = df_sorted.drop(columns=['date_datetime_check'])
            df = df_sorted
        
        df.to_sql('copper_data', conn, if_exists='replace', index=False)
        
        conn.commit()
        conn.close()
        print(f"تم إعادة ترتيب جميع البيانات في جدول copper_data - عدد الصفوف: {len(df)}")
        if 'date' in df.columns and len(df) > 0:
            print(f"أول تاريخ في قاعدة البيانات: {df['date'].iloc[0]}, آخر تاريخ: {df['date'].iloc[-1]}")
    except Exception as e:
        print(f"خطأ في إعادة ترتيب البيانات: {e}")
        import traceback
        traceback.print_exc()

# ---------------------------------------------------------
# دالة لحفظ التوقع في قاعدة البيانات
# ---------------------------------------------------------
def save_prediction_to_db(features_dict, predicted_price):
    """حفظ التوقع في جدول predictions"""
    try:
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        datetime_str = now.strftime('%Y-%m-%d %H:%M:%S')
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                datetime, date, time,
                global_demand_index, oil_price, usd_index,
                china_industry_output, energy_cost_index,
                market_sentiment, supply_disruption_index,
                copper_price, predicted_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime_str, date_str, time_str,
            features_dict['global_demand_index'],
            features_dict['oil_price'],
            features_dict['usd_index'],
            features_dict['china_industry_output'],
            features_dict['energy_cost_index'],
            features_dict['market_sentiment'],
            features_dict['supply_disruption_index'],
            features_dict['copper_price'],
            float(predicted_price)
        ))
        
        conn.commit()
        conn.close()
        print(f"تم حفظ التوقع في قاعدة البيانات: {datetime_str}")
    except Exception as e:
        print(f"خطأ في حفظ التوقع: {e}")


# ---------------------------------------------------------
# دالة لحفظ التوقع في جدول copper_data مع ترتيب البيانات
# ---------------------------------------------------------
def save_prediction_to_copper_data(features_dict, predicted_price):
    """حفظ التوقع كسطر جديد في جدول copper_data مع ترتيب البيانات حسب التاريخ"""
    try:
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(DB_FILE)
        
        # قراءة البيانات الحالية
        df = pd.read_sql_query("SELECT * FROM copper_data", conn)
        
        # إنشاء سطر جديد
        new_row = {
            'date': date_str,
            'global_demand_index': features_dict['global_demand_index'],
            'oil_price': features_dict['oil_price'],
            'usd_index': features_dict['usd_index'],
            'china_industry_output': features_dict['china_industry_output'],
            'energy_cost_index': features_dict['energy_cost_index'],
            'market_sentiment': features_dict['market_sentiment'],
            'supply_disruption_index': features_dict['supply_disruption_index'],
            'copper_price': features_dict['copper_price'],
            'Next_Day_Copper_Price': float(predicted_price)
        }
        
        # إذا كان التاريخ موجوداً مسبقاً، نستبدل السطر القديم
        if 'date' in df.columns and not df.empty:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            date_to_check = pd.to_datetime(date_str, errors='coerce')
            # إزالة الصفوف التي لها نفس التاريخ
            df = df[df['date'] != date_to_check]
            # تحويل التاريخ مرة أخرى إلى string
            df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
        
        # إضافة السطر الجديد إلى DataFrame
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        
        # ترتيب البيانات حسب التاريخ (تنازلي - من الأحدث إلى الأقدم) ثم حفظها
        if 'date' in df.columns:
            # إنشاء عمود مؤقت للترتيب
            df['date_datetime'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            
            # فصل البيانات الصالحة عن غير الصالحة
            df_valid = df[df['date_datetime'].notna()].copy()
            df_invalid = df[df['date_datetime'].isna()].copy()
            
            # ترتيب البيانات الصالحة حسب التاريخ (تنازلي - من الأحدث إلى الأقدم)
            if not df_valid.empty:
                df_valid = df_valid.sort_values('date_datetime', ascending=False, na_position='last', kind='mergesort')
                df_valid = df_valid.reset_index(drop=True)
                # تحويل التاريخ إلى string بعد الترتيب
                df_valid['date'] = df_valid['date_datetime'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
            
            # دمج البيانات الصالحة وغير الصالحة (الصالحة أولاً - مرتبة تنازلياً)
            if not df_invalid.empty:
                df_invalid = df_invalid.reset_index(drop=True)
                df = pd.concat([df_valid, df_invalid], ignore_index=True)
            else:
                df = df_valid
            
            # إزالة العمود المؤقت
            if 'date_datetime' in df.columns:
                df = df.drop(columns=['date_datetime'])
        
        # تنظيف البيانات قبل الحفظ (إزالة المكررات بناءً على التاريخ فقط)
        if 'date' in df.columns:
            df = df.drop_duplicates(subset=['date'], keep='last')
        # إزالة الصفوف التي لا تحتوي على تاريخ فقط (وليس جميع الأعمدة)
        if 'date' in df.columns:
            df = df[df['date'].notna()]
        df = df.reset_index(drop=True)
        
        # حفظ البيانات المرتبة مرة أخرى في قاعدة البيانات
        df.to_sql('copper_data', conn, if_exists='replace', index=False)
        
        conn.commit()
        conn.close()
        print(f"تم حفظ التوقع في جدول copper_data: {date_str} - السعر المتوقع: {predicted_price}")
    except Exception as e:
        print(f"خطأ في حفظ التوقع في copper_data: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------
# دالة لتحديث قاعدة البيانات تلقائياً
# ---------------------------------------------------------
def auto_update_database():
    """تحديث قاعدة البيانات تلقائياً كل دقيقة"""
    while True:
        try:
            time.sleep(60)  # انتظار دقيقة واحدة
            
            # جلب البيانات الحقيقية
            oil_price = get_brent_price()
            usd_index = get_dxy_value()
            news_sentiment = get_market_sentiment_from_news()
            
            df = load_data_from_db()
            df = clean_dataframe(df)
            
            if len(df) == 0:
                continue
            
            # الحصول على آخر سعر نحاس
            last_copper_price = df['copper_price'].iloc[-1] if len(df) > 0 else 10000.0
            
            # حساب المتغيرات الأخرى
            other = df[[col for col in feature_columns if col not in ["oil_price", "usd_index", "market_sentiment", "copper_price"]]].mean().values.tolist()
            
            full_features = [
                other[0],  # global_demand_index
                oil_price,
                usd_index,
                other[1],  # china_industry_output
                other[2],  # energy_cost_index
                news_sentiment,
                other[3],  # supply_disruption_index
                last_copper_price  # copper_price
            ]
            
            # إجراء التوقع
            scaled = scaler.transform([full_features])
            prediction = model.predict(scaled)[0]
            
            features_dict = {
                'global_demand_index': full_features[0],
                'oil_price': full_features[1],
                'usd_index': full_features[2],
                'china_industry_output': full_features[3],
                'energy_cost_index': full_features[4],
                'market_sentiment': full_features[5],
                'supply_disruption_index': full_features[6],
                'copper_price': full_features[7]
            }
            
            # حفظ التوقع في قاعدة البيانات
            save_prediction_to_db(features_dict, prediction)
            
            print(f"تم التحديث التلقائي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"خطأ في التحديث التلقائي: {e}")

# متغير لتتبع حالة التحديث التلقائي
auto_update_started = False

def start_auto_update():
    """بدء التحديث التلقائي"""
    global auto_update_started
    if not auto_update_started:
        update_thread = threading.Thread(target=auto_update_database, daemon=True)
        update_thread.start()
        auto_update_started = True
        print("تم بدء التحديث التلقائي لقاعدة البيانات كل دقيقة")

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
    df = load_data_from_db()
    df = clean_dataframe(df)  # تنظيف البيانات

    # ترتيب البيانات حسب التاريخ (تنازلي - من الأحدث إلى الأقدم) للتأكد
    if 'date' in df.columns and not df.empty:
        df['date_datetime'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        df_valid = df[df['date_datetime'].notna()].copy()
        df_invalid = df[df['date_datetime'].isna()].copy()
        
        if not df_valid.empty:
            df_valid = df_valid.sort_values('date_datetime', ascending=False, na_position='last', kind='mergesort')
            df_valid = df_valid.reset_index(drop=True)
            df_valid['date'] = df_valid['date_datetime'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
        
        if not df_invalid.empty:
            df_invalid = df_invalid.reset_index(drop=True)
            df = pd.concat([df_valid, df_invalid], ignore_index=True)
        else:
            df = df_valid
        
        if 'date_datetime' in df.columns:
            df = df.drop(columns=['date_datetime'])

    # آخر 60 يوم (أول 60 سطر في البيانات المرتبة تنازلياً)
    if 'date' in df.columns:
        history_dates = df['date'].head(60).tolist()
    else:
        history_dates = list(range(len(df.head(60))))

    history_prices = df['copper_price'].head(60).tolist()

    oil_price = get_brent_price()
    usd_index = get_dxy_value()

    try:
        news_sentiment = get_market_sentiment_from_news()
        sentiment = float(news_sentiment) if news_sentiment is not None else (oil_price / 100) - (usd_index / 200)
    except Exception as e:
        print("Error getting news sentiment in dashboard_data:", e)
        sentiment = (oil_price / 100) - (usd_index / 200)

    # الحصول على آخر سعر نحاس من البيانات (أول سطر في البيانات المرتبة تنازلياً)
    last_copper_price = df['copper_price'].iloc[0] if len(df) > 0 else 10000.0
    
    # إعداد الميزات مع إضافة copper_price
    last_row_features = df[feature_columns[:-1]].head(1).values[0].tolist()  # جميع الميزات ما عدا copper_price
    last_row_features.append(last_copper_price)  # إضافة copper_price
    last_scaled = scaler.transform([last_row_features])
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
        float(data['supply_disruption_index']),
        float(data['copper_price'])  # إضافة copper_price كخاصية
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
        df = load_data_from_db()
        df = clean_dataframe(df)  # تنظيف البيانات

        if not set(feature_columns).issubset(df.columns) or 'Next_Day_Copper_Price' not in df.columns:
            return jsonify({'success': False, 'error': 'Required columns missing in dataset'}), 400

        X = df[feature_columns].values
        y = df['Next_Day_Copper_Price'].values  # استخدام Next_Day_Copper_Price كهدف

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
        df = load_data_from_db()
        df = clean_dataframe(df)  # تنظيف البيانات

        oil_price = get_brent_price()
        usd_index = get_dxy_value()
        news_sentiment = get_market_sentiment_from_news()
        
        # الحصول على آخر سعر نحاس من البيانات
        last_copper_price = df['copper_price'].iloc[-1] if len(df) > 0 else 10000.0

        other = df[[col for col in feature_columns if col not in ["oil_price", "usd_index", "market_sentiment", "copper_price"]]].mean().values.tolist()

        full_features = [
            other[0],  # global_demand_index
            oil_price,
            usd_index,
            other[1],  # china_industry_output
            other[2],  # energy_cost_index
            news_sentiment,
            other[3],  # supply_disruption_index
            last_copper_price  # copper_price
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
            'supply_disruption_index': full_features[6],
            'copper_price': full_features[7]
        }

        # حفظ التوقع في قاعدة البيانات
        save_prediction_to_db(features_dict, prediction)
        
        # حفظ التوقع في جدول copper_data مع ترتيب البيانات
        save_prediction_to_copper_data(features_dict, prediction)

        # Return both top-level keys and a `features_used` object for the frontend
        return jsonify({
            'predicted_copper_price': float(prediction),
            'oil_price': features_dict['oil_price'],
            'usd_index': features_dict['usd_index'],
            'market_sentiment': features_dict['market_sentiment'],
            'features_used': features_dict,
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

        # read existing data from SQLite and concat
        existing_df = load_data_from_db()
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = clean_dataframe(combined)

        # ترتيب البيانات حسب التاريخ (تنازلي - من الأحدث إلى الأقدم)
        if 'date' in combined.columns and not combined.empty:
            combined['date_datetime'] = pd.to_datetime(combined['date'], errors='coerce', infer_datetime_format=True)
            df_valid = combined[combined['date_datetime'].notna()].copy()
            df_invalid = combined[combined['date_datetime'].isna()].copy()
            
            if not df_valid.empty:
                df_valid = df_valid.sort_values('date_datetime', ascending=False, na_position='last', kind='mergesort')
                df_valid = df_valid.reset_index(drop=True)
                df_valid['date'] = df_valid['date_datetime'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
            
            if not df_invalid.empty:
                df_invalid = df_invalid.reset_index(drop=True)
                combined = pd.concat([df_valid, df_invalid], ignore_index=True)
            else:
                combined = df_valid
            
            if 'date_datetime' in combined.columns:
                combined = combined.drop(columns=['date_datetime'])
            
            combined = combined.drop_duplicates(subset=['date'], keep='last')
            combined = combined[combined['date'].notna()]
            combined = combined.reset_index(drop=True)

        # save back to SQLite
        conn = sqlite3.connect(DB_FILE)
        combined.to_sql('copper_data', conn, if_exists='replace', index=False)
        conn.close()

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
# صفحة عرض قاعدة البيانات
# ---------------------------------------------------------
@app.route('/database')
def database_view():
    """عرض صفحة قاعدة البيانات"""
    return render_template('database.html')


@app.route('/database-data')
def database_data():
    """API لجلب البيانات من قاعدة البيانات مع pagination"""
    try:
        # إعادة ترتيب جميع البيانات في قاعدة البيانات عند كل تحديث
        reorder_copper_data()
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        search = request.args.get('search', '', type=str)
        sort_column = request.args.get('sort', 'date', type=str)
        sort_order = request.args.get('order', 'desc', type=str)  # افتراضياً من الأحدث إلى الأقدم
        
        conn = sqlite3.connect(DB_FILE)
        
        # بناء استعلام البحث
        where_clause = ""
        params = []
        if search:
            where_clause = "WHERE date LIKE ? OR copper_price LIKE ? OR Next_Day_Copper_Price LIKE ?"
            search_pattern = f"%{search}%"
            params = [search_pattern, search_pattern, search_pattern]
        
        # التحقق من أن عمود الترتيب موجود
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(copper_data)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if sort_column not in columns:
            sort_column = 'date'
        
        # إذا كان الترتيب حسب التاريخ، نستخدم ترتيب تاريخي صحيح
        if sort_column == 'date':
            # جلب جميع البيانات أولاً للترتيب الصحيح حسب التاريخ
            query_all = f"SELECT * FROM copper_data {where_clause}"
            cursor.execute(query_all, params)
            rows_all = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            df = pd.DataFrame(rows_all, columns=column_names)
            
            # تحويل عمود date إلى datetime وترتيب البيانات
            if 'date' in df.columns and not df.empty:
                # إنشاء عمود مؤقت للترتيب
                # استخدام infer_datetime_format للتعرف التلقائي على التنسيق
                df['date_datetime'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
                
                # إزالة الصفوف التي تحتوي على تواريخ غير صالحة قبل الترتيب
                df_valid = df[df['date_datetime'].notna()].copy()
                df_invalid = df[df['date_datetime'].isna()].copy()
                
                # ترتيب البيانات الصالحة حسب التاريخ (سنة، شهر، يوم)
                if not df_valid.empty:
                    ascending_order = (sort_order.lower() == 'asc')
                    # ترتيب حسب التاريخ مع إعادة ضبط الفهرس
                    # استخدام mergesort للترتيب المستقر
                    df_valid = df_valid.sort_values('date_datetime', ascending=ascending_order, na_position='last', kind='mergesort')
                    df_valid = df_valid.reset_index(drop=True)
                    # تحويل التاريخ إلى string بتنسيق ISO (YYYY-MM-DD) للبيانات الصالحة
                    df_valid['date'] = df_valid['date_datetime'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
                    
                    # طباعة أول وآخر تاريخ للتحقق (للتطوير فقط)
                    if len(df_valid) > 0:
                        print(f"[database_data] الترتيب: {'تصاعدي' if ascending_order else 'تنازلي'}, أول تاريخ: {df_valid['date'].iloc[0]}, آخر تاريخ: {df_valid['date'].iloc[-1]}, عدد الصفوف: {len(df_valid)}")
                
                # للبيانات غير الصالحة، نحتفظ بالتاريخ الأصلي كما هو
                if not df_invalid.empty:
                    df_invalid = df_invalid.reset_index(drop=True)
                    # التاريخ الأصلي موجود بالفعل في عمود 'date'، لا نحتاج لتغييره
                
                # دمج البيانات الصالحة وغير الصالحة
                # عند الترتيب التصاعدي: البيانات الصالحة أولاً (مرتبة تصاعدياً)، ثم البيانات غير الصالحة
                # عند الترتيب التنازلي: البيانات الصالحة أولاً (مرتبة تنازلياً)، ثم البيانات غير الصالحة
                if not df_invalid.empty:
                    if ascending_order:
                        # عند الترتيب التصاعدي، البيانات غير الصالحة في النهاية
                        df = pd.concat([df_valid, df_invalid], ignore_index=True)
                    else:
                        # عند الترتيب التنازلي، البيانات غير الصالحة في النهاية أيضاً
                        df = pd.concat([df_valid, df_invalid], ignore_index=True)
                else:
                    df = df_valid
                
                # إزالة العمود المؤقت
                if 'date_datetime' in df.columns:
                    df = df.drop(columns=['date_datetime'])
            
            # حساب العدد الإجمالي
            total = len(df)
            
            # تطبيق pagination
            offset = (page - 1) * per_page
            df = df.iloc[offset:offset + per_page]
        else:
            # للترتيب حسب أعمدة أخرى، نستخدم SQL
            order_by = f"ORDER BY {sort_column} {sort_order.upper()}"
            
            # حساب العدد الإجمالي
            count_query = f"SELECT COUNT(*) as total FROM copper_data {where_clause}"
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            # جلب البيانات مع pagination
            offset = (page - 1) * per_page
            query = f"SELECT * FROM copper_data {where_clause} {order_by} LIMIT ? OFFSET ?"
            params_with_limit = params + [per_page, offset]
            
            cursor.execute(query, params_with_limit)
            rows = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            df = pd.DataFrame(rows, columns=column_names)
            
            # تحويل عمود date إلى كائن date
            if 'date' in df.columns and not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
        
        conn.close()
        
        # تحويل البيانات إلى قائمة
        data = df.to_dict('records')
        
        # تنسيق الأرقام وتنظيف القيم NaN
        for row in data:
            for key, value in row.items():
                if key == 'date':
                    # التأكد من أن التاريخ ليس NaN أو None
                    if value is None:
                        row[key] = None
                    elif pd.isna(value):
                        row[key] = None
                    elif isinstance(value, str):
                        # إذا كان التاريخ string، نحتفظ به كما هو (حتى لو كان فارغاً)
                        if value.strip() == '' or value == 'NaT' or value == 'nan':
                            row[key] = None
                        else:
                            row[key] = value
                    else:
                        # إذا كان التاريخ نوع آخر، نحاول تحويله إلى string
                        row[key] = str(value) if value is not None else None
                elif isinstance(value, (int, float)) and key != 'date':
                    if pd.isna(value):
                        row[key] = None
                    else:
                        row[key] = round(float(value), 2) if value is not None else None
                elif pd.isna(value):
                    row[key] = None
        
        return jsonify({
            'success': True,
            'data': data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': math.ceil(total / per_page) if per_page > 0 else 0
            }
        })
    except Exception as e:
        print('Error in database_data:', e)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predictions-data')
def predictions_data():
    """API لجلب التوقعات المحفوظة من قاعدة البيانات"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        search = request.args.get('search', '', type=str)
        sort_column = request.args.get('sort', 'date', type=str)
        sort_order = request.args.get('order', 'desc', type=str)  # افتراضياً من الأحدث إلى الأقدم
        
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # بناء استعلام البحث
        where_clause = ""
        params = []
        if search:
            where_clause = "WHERE datetime LIKE ? OR date LIKE ? OR predicted_price LIKE ?"
            search_pattern = f"%{search}%"
            params = [search_pattern, search_pattern, search_pattern]
        
        # التحقق من أن عمود الترتيب موجود
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if sort_column not in columns:
            sort_column = 'date'
        
        # إذا كان الترتيب حسب التاريخ، نستخدم ترتيب تاريخي صحيح
        if sort_column == 'date':
            # جلب جميع البيانات أولاً للترتيب الصحيح حسب التاريخ
            query_all = f"SELECT * FROM predictions {where_clause}"
            cursor.execute(query_all, params)
            rows_all = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            df = pd.DataFrame(rows_all, columns=column_names)
            
            # تحويل عمود date إلى datetime وترتيب البيانات
            if 'date' in df.columns and not df.empty:
                # تحويل التاريخ إلى datetime مع معالجة الأخطاء (محاولة تنسيقات متعددة)
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
                # إذا فشل التنسيق الأول، جرب تنسيقات أخرى
                if df['date'].isna().any():
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # إزالة الصفوف التي تحتوي على تواريخ غير صالحة قبل الترتيب
                df_valid = df[df['date'].notna()].copy()
                df_invalid = df[df['date'].isna()].copy()
                
                # ترتيب البيانات الصالحة حسب التاريخ (سنة، شهر، يوم)
                if not df_valid.empty:
                    ascending_order = (sort_order.lower() == 'asc')
                    # ترتيب حسب التاريخ مع إعادة ضبط الفهرس
                    df_valid = df_valid.sort_values('date', ascending=ascending_order, na_position='last', kind='mergesort')
                    df_valid = df_valid.reset_index(drop=True)
                
                # دمج البيانات الصالحة وغير الصالحة (الصالحة أولاً)
                if not df_invalid.empty:
                    df_invalid = df_invalid.reset_index(drop=True)
                    df = pd.concat([df_valid, df_invalid], ignore_index=True)
                else:
                    df = df_valid
                
                # تحويل التاريخ إلى string بتنسيق ISO (YYYY-MM-DD)
                df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
            
            # حساب العدد الإجمالي
            total = len(df)
            
            # تطبيق pagination
            offset = (page - 1) * per_page
            df = df.iloc[offset:offset + per_page]
        else:
            # للترتيب حسب أعمدة أخرى، نستخدم SQL
            order_by = f"ORDER BY {sort_column} {sort_order.upper()}"
            
            # حساب العدد الإجمالي
            count_query = f"SELECT COUNT(*) as total FROM predictions {where_clause}"
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            # جلب البيانات مع pagination
            offset = (page - 1) * per_page
            query = f"SELECT * FROM predictions {where_clause} {order_by} LIMIT ? OFFSET ?"
            params_with_limit = params + [per_page, offset]
            
            cursor.execute(query, params_with_limit)
            rows = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            df = pd.DataFrame(rows, columns=column_names)
            
            # تحويل عمود date إلى كائن date
            if 'date' in df.columns and not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
        
        conn.close()
        
        # تحويل البيانات إلى قائمة
        data = df.to_dict('records')
        
        # تنسيق الأرقام وتنظيف القيم NaN
        for row in data:
            for key, value in row.items():
                if key == 'date':
                    # التأكد من أن التاريخ ليس NaN
                    if pd.isna(value) or value == 'NaT' or value == 'nan' or value is None:
                        row[key] = None
                    elif isinstance(value, str) and (value == 'NaT' or value == 'nan'):
                        row[key] = None
                elif isinstance(value, (int, float)) and key not in ['id', 'datetime', 'date', 'time', 'created_at']:
                    if pd.isna(value):
                        row[key] = None
                    else:
                        row[key] = round(float(value), 2) if value is not None else None
                elif pd.isna(value):
                    row[key] = None
        
        return jsonify({
            'success': True,
            'data': data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': math.ceil(total / per_page) if per_page > 0 else 0
            }
        })
    except Exception as e:
        print('Error in predictions_data:', e)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


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



# إعداد جدول predictions عند تحميل الوحدة
init_predictions_table()

if __name__ == '__main__':
    # بدء التحديث التلقائي
    start_auto_update()
    
    # تشغيل التطبيق محليًا عند استدعاء الملف مباشرة
    # ضبط debug=True مفيد أثناء التطوير، يمكنك تغييره إلى False في الإنتاج
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    # عند استخدام gunicorn أو خادم آخر
    start_auto_update()

    

