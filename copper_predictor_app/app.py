from flask import Flask ,request ,render_template ,jsonify 
import pickle 
import numpy as np 
import pandas as pd 
from pathlib import Path 
import requests 
import yfinance as yf 
from sklearn .metrics import mean_squared_error ,mean_absolute_error 
import math 
import sqlite3 
from datetime import datetime ,timedelta 
import threading 
import time 

app =Flask (__name__ )


with open ('model/copper_model.pkl','rb')as model_file :
    model ,scaler =pickle .load (model_file )


DB_FILE =Path (__file__ ).resolve ().parent /'data'/'copper_data.db'


feature_columns =[
'global_demand_index',
'oil_price',
'usd_index',
'china_industry_output',
'energy_cost_index',
'market_sentiment',
'supply_disruption_index',
'copper_price'
]


OIL_API_KEY ="980ecfe16b13b1881b03de30115dbb59897c0da5c5333e30717343e244ad7927"





def init_predictions_table ():
    conn =sqlite3 .connect (DB_FILE )
    cursor =conn .cursor ()
    cursor .execute ('''
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
    conn .commit ()
    conn .close ()




def load_data_from_db ():
    conn =sqlite3 .connect (DB_FILE )
    df =pd .read_sql_query ("SELECT * FROM copper_data",conn )
    conn .close ()
    return df 

def reorder_copper_data ():
    try :
        conn =sqlite3 .connect (DB_FILE )


        df =pd .read_sql_query ("SELECT * FROM copper_data",conn )

        if df .empty or 'date'not in df .columns :
            conn .close ()
            return 




        df ['date_datetime']=pd .to_datetime (df ['date'],errors ='coerce',infer_datetime_format =True )


        df_valid =df [df ['date_datetime'].notna ()].copy ()
        df_invalid =df [df ['date_datetime'].isna ()].copy ()


        if not df_valid .empty :
            df_valid =df_valid .sort_values ('date_datetime',ascending =False ,na_position ='last',kind ='mergesort')
            df_valid =df_valid .reset_index (drop =True )

            df_valid ['date']=df_valid ['date_datetime'].apply (lambda x :x .strftime ('%Y-%m-%d')if pd .notna (x )else None )


            if len (df_valid )>0 :
                print (f"[reorder_copper_data] أول تاريخ بعد الترتيب: {df_valid ['date'].iloc [0 ]}, آخر تاريخ: {df_valid ['date'].iloc [-1 ]}, عدد الصفوف: {len (df_valid )}")



        if not df_invalid .empty :
            df_invalid =df_invalid .reset_index (drop =True )

            df =pd .concat ([df_valid ,df_invalid ],ignore_index =True )
        else :
            df =df_valid 


        if 'date_datetime'in df .columns :
            df =df .drop (columns =['date_datetime'])


        if 'date'in df .columns :
            df =df .drop_duplicates (subset =['date'],keep ='last')
        if 'date'in df .columns :
            df =df [df ['date'].notna ()]
        df =df .reset_index (drop =True )



        if 'date'in df .columns and not df .empty :

            df ['date_datetime_check']=pd .to_datetime (df ['date'],errors ='coerce',infer_datetime_format =True )
            df_sorted =df .sort_values ('date_datetime_check',ascending =False ,na_position ='last',kind ='mergesort')
            df_sorted =df_sorted .reset_index (drop =True )
            df_sorted =df_sorted .drop (columns =['date_datetime_check'])
            df =df_sorted 

        df .to_sql ('copper_data',conn ,if_exists ='replace',index =False )

        conn .commit ()
        conn .close ()
        print (f"تم إعادة ترتيب جميع البيانات في جدول copper_data - عدد الصفوف: {len (df )}")
        if 'date'in df .columns and len (df )>0 :
            print (f"أول تاريخ في قاعدة البيانات: {df ['date'].iloc [0 ]}, آخر تاريخ: {df ['date'].iloc [-1 ]}")
    except Exception as e :
        print (f"خطأ في إعادة ترتيب البيانات: {e }")
        import traceback 
        traceback .print_exc ()

def save_prediction_to_db (features_dict ,predicted_price ):
    try :
        now =datetime .now ()
        date_str =now .strftime ('%Y-%m-%d')
        time_str =now .strftime ('%H:%M:%S')
        datetime_str =now .strftime ('%Y-%m-%d %H:%M:%S')

        conn =sqlite3 .connect (DB_FILE )
        cursor =conn .cursor ()

        cursor .execute ('''
            INSERT INTO predictions (
                datetime, date, time,
                global_demand_index, oil_price, usd_index,
                china_industry_output, energy_cost_index,
                market_sentiment, supply_disruption_index,
                copper_price, predicted_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',(
        datetime_str ,date_str ,time_str ,
        features_dict ['global_demand_index'],
        features_dict ['oil_price'],
        features_dict ['usd_index'],
        features_dict ['china_industry_output'],
        features_dict ['energy_cost_index'],
        features_dict ['market_sentiment'],
        features_dict ['supply_disruption_index'],
        features_dict ['copper_price'],
        float (predicted_price )
        ))

        conn .commit ()
        conn .close ()
        print (f"تم حفظ التوقع في قاعدة البيانات: {datetime_str }")
    except Exception as e :
        print (f"خطأ في حفظ التوقع: {e }")

def save_prediction_to_copper_data (features_dict ,predicted_price ):
    try :
        now =datetime .now ()
        date_str =now .strftime ('%Y-%m-%d')

        conn =sqlite3 .connect (DB_FILE )


        df =pd .read_sql_query ("SELECT * FROM copper_data",conn )


        new_row ={
        'date':date_str ,
        'global_demand_index':features_dict ['global_demand_index'],
        'oil_price':features_dict ['oil_price'],
        'usd_index':features_dict ['usd_index'],
        'china_industry_output':features_dict ['china_industry_output'],
        'energy_cost_index':features_dict ['energy_cost_index'],
        'market_sentiment':features_dict ['market_sentiment'],
        'supply_disruption_index':features_dict ['supply_disruption_index'],
        'copper_price':features_dict ['copper_price'],
        'Next_Day_Copper_Price':float (predicted_price )
        }


        if 'date'in df .columns and not df .empty :
            df ['date']=pd .to_datetime (df ['date'],format ='%Y-%m-%d',errors ='coerce')
            date_to_check =pd .to_datetime (date_str ,errors ='coerce')

            df =df [df ['date']!=date_to_check ]

            df ['date']=df ['date'].apply (lambda x :x .strftime ('%Y-%m-%d')if pd .notna (x )else None )


        new_df =pd .DataFrame ([new_row ])
        df =pd .concat ([df ,new_df ],ignore_index =True )


        if 'date'in df .columns :

            df ['date_datetime']=pd .to_datetime (df ['date'],format ='%Y-%m-%d',errors ='coerce')


            df_valid =df [df ['date_datetime'].notna ()].copy ()
            df_invalid =df [df ['date_datetime'].isna ()].copy ()


            if not df_valid .empty :
                df_valid =df_valid .sort_values ('date_datetime',ascending =False ,na_position ='last',kind ='mergesort')
                df_valid =df_valid .reset_index (drop =True )

                df_valid ['date']=df_valid ['date_datetime'].apply (lambda x :x .strftime ('%Y-%m-%d')if pd .notna (x )else None )


            if not df_invalid .empty :
                df_invalid =df_invalid .reset_index (drop =True )
                df =pd .concat ([df_valid ,df_invalid ],ignore_index =True )
            else :
                df =df_valid 


            if 'date_datetime'in df .columns :
                df =df .drop (columns =['date_datetime'])


        if 'date'in df .columns :
            df =df .drop_duplicates (subset =['date'],keep ='last')

        if 'date'in df .columns :
            df =df [df ['date'].notna ()]
        df =df .reset_index (drop =True )


        df .to_sql ('copper_data',conn ,if_exists ='replace',index =False )

        conn .commit ()
        conn .close ()
        print (f"تم حفظ التوقع في جدول copper_data: {date_str } - السعر المتوقع: {predicted_price }")
    except Exception as e :
        print (f"خطأ في حفظ التوقع في copper_data: {e }")
        import traceback 
        traceback .print_exc ()

def auto_update_database ():
    while True :
        try :
            time .sleep (60 )


            oil_price =get_brent_price ()
            usd_index =get_dxy_value ()
            news_sentiment =get_market_sentiment_from_news ()

            df =load_data_from_db ()
            df =clean_dataframe (df )

            if len (df )==0 :
                continue 


            last_copper_price =df ['copper_price'].iloc [-1 ]if len (df )>0 else 10000.0 


            other =df [[col for col in feature_columns if col not in ["oil_price","usd_index","market_sentiment","copper_price"]]].mean ().values .tolist ()

            full_features =[
            other [0 ],
            oil_price ,
            usd_index ,
            other [1 ],
            other [2 ],
            news_sentiment ,
            other [3 ],
            last_copper_price 
            ]


            scaled =scaler .transform ([full_features ])
            prediction =model .predict (scaled )[0 ]

            features_dict ={
            'global_demand_index':full_features [0 ],
            'oil_price':full_features [1 ],
            'usd_index':full_features [2 ],
            'china_industry_output':full_features [3 ],
            'energy_cost_index':full_features [4 ],
            'market_sentiment':full_features [5 ],
            'supply_disruption_index':full_features [6 ],
            'copper_price':full_features [7 ]
            }


            save_prediction_to_db (features_dict ,prediction )

            print (f"تم التحديث التلقائي: {datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')}")

        except Exception as e :
            print (f"خطأ في التحديث التلقائي: {e }")

auto_update_started =False 

def start_auto_update ():
    global auto_update_started 
    if not auto_update_started :
        update_thread =threading .Thread (target =auto_update_database ,daemon =True )
        update_thread .start ()
        auto_update_started =True 
        print ("تم بدء التحديث التلقائي لقاعدة البيانات كل دقيقة")

def clean_dataframe (df :pd .DataFrame )->pd .DataFrame :

    df =df .drop_duplicates ()

    df =df .dropna ()

    df =df .reset_index (drop =True )
    return df 

def get_brent_price ():
    try :
        url ="https://api.oilpriceapi.com/v1/prices/latest"
        headers ={"Authorization":f"Token {OIL_API_KEY }"}
        response =requests .get (url ,headers =headers )
        data =response .json ()
        return float (data ["data"]["price"])
    except Exception as e :
        print ("Error fetching Brent:",e )
        return 75.0 

def get_dxy_value ():
    try :
        ticker =yf .Ticker ("DX-Y.NYB")
        data =ticker .history (period ="1d")
        if data .empty :
            return 102.0 
        return float (data ["Close"].iloc [-1 ])
    except Exception as e :
        print ("Error fetching DXY:",e )
        return 102.0 

@app .route ('/dashboard')
def dashboard ():
    return render_template ('index.html')

@app .route ('/dashboard-data')
def dashboard_data ():
    df =load_data_from_db ()
    df =clean_dataframe (df )


    if 'date'in df .columns and not df .empty :
        df ['date_datetime']=pd .to_datetime (df ['date'],errors ='coerce',infer_datetime_format =True )
        df_valid =df [df ['date_datetime'].notna ()].copy ()
        df_invalid =df [df ['date_datetime'].isna ()].copy ()

        if not df_valid .empty :
            df_valid =df_valid .sort_values ('date_datetime',ascending =False ,na_position ='last',kind ='mergesort')
            df_valid =df_valid .reset_index (drop =True )
            df_valid ['date']=df_valid ['date_datetime'].apply (lambda x :x .strftime ('%Y-%m-%d')if pd .notna (x )else None )

        if not df_invalid .empty :
            df_invalid =df_invalid .reset_index (drop =True )
            df =pd .concat ([df_valid ,df_invalid ],ignore_index =True )
        else :
            df =df_valid 

        if 'date_datetime'in df .columns :
            df =df .drop (columns =['date_datetime'])


    if 'date'in df .columns :
        history_dates =df ['date'].head (60 ).tolist ()
    else :
        history_dates =list (range (len (df .head (60 ))))

    history_prices =df ['copper_price'].head (60 ).tolist ()

    oil_price =get_brent_price ()
    usd_index =get_dxy_value ()

    try :
        news_sentiment =get_market_sentiment_from_news ()
        sentiment =float (news_sentiment )if news_sentiment is not None else (oil_price /100 )-(usd_index /200 )
    except Exception as e :
        print ("Error getting news sentiment in dashboard_data:",e )
        sentiment =(oil_price /100 )-(usd_index /200 )


    last_copper_price =df ['copper_price'].iloc [0 ]if len (df )>0 else 10000.0 


    last_row_features =df [feature_columns [:-1 ]].head (1 ).values [0 ].tolist ()
    last_row_features .append (last_copper_price )
    last_scaled =scaler .transform ([last_row_features ])
    predicted_price =model .predict (last_scaled )[0 ]

    return jsonify ({
    "predicted_price":predicted_price ,
    "oil_price":oil_price ,
    "usd_index":usd_index ,
    "market_sentiment":sentiment ,
    "history_dates":history_dates ,
    "history_prices":history_prices 
    })

@app .route ('/predict',methods =['GET','POST'])
def predict ():
    if request .method =='GET':
        return render_template ('predict.html')

    data =request .get_json (force =True )

    features =[
    float (data ['global_demand_index']),
    float (data ['oil_price']),
    float (data ['usd_index']),
    float (data ['china_industry_output']),
    float (data ['energy_cost_index']),
    float (data ['market_sentiment']),
    float (data ['supply_disruption_index']),
    float (data ['copper_price'])
    ]

    scaled =scaler .transform ([features ])
    prediction =model .predict (scaled )[0 ]
    coefficients =model .coef_ .tolist ()

    return jsonify ({
    "predicted_copper_price":prediction ,
    "coefficients":coefficients 
    })

@app .route ('/manual')
def manual ():
    return render_template ('predict.html')

@app .route ("/auto")
def auto_page ():
    return render_template ("api_predict.html")

@app .route ('/model_info')
def model_info_page ():
    return render_template ('model_info.html')

@app .route ('/model-info-data')
def model_info_data ():
    try :
        df =load_data_from_db ()
        df =clean_dataframe (df )

        if not set (feature_columns ).issubset (df .columns )or 'Next_Day_Copper_Price'not in df .columns :
            return jsonify ({'success':False ,'error':'Required columns missing in dataset'}),400 

        X =df [feature_columns ].values 
        y =df ['Next_Day_Copper_Price'].values 

        X_scaled =scaler .transform (X )
        preds =model .predict (X_scaled )

        mse =mean_squared_error (y ,preds )
        rmse =math .sqrt (mse )
        mae =mean_absolute_error (y ,preds )
        with np .errstate (divide ='ignore',invalid ='ignore'):
            mape =np .mean (np .abs ((y -preds )/np .where (y ==0 ,np .nan ,y )))*100 
            if np .isnan (mape ):
                mape =None 

        summary ={
        'rows':int (df .shape [0 ]),
        'columns':int (df .shape [1 ]),
        'target_mean':float (np .nanmean (y )),
        'target_std':float (np .nanstd (y ))
        }

        coefficients =getattr (model ,'coef_',None )
        intercept =getattr (model ,'intercept_',None )

        return jsonify ({
        'success':True ,
        'metrics':{
        'mae':float (mae ),
        'mse':float (mse ),
        'rmse':float (rmse ),
        'mape':None if mape is None else float (mape )
        },
        'summary':summary ,
        'coefficients':coefficients .tolist ()if coefficients is not None else None ,
        'intercept':float (intercept )if intercept is not None else None ,
        'feature_columns':feature_columns 
        })
    except Exception as e :
        print ('Error in model_info_data:',e )
        return jsonify ({'success':False ,'error':str (e )}),500 

def get_copper_price_from_yfinance ():
    try :
        # عقد النحاس الآجل في COMEX
        ticker =yf .Ticker ("HG=F")
        data =ticker .history (period ="1d")
        if data .empty :
            # في حال عدم توفر بيانات من ياهو، نستخدم قيمة افتراضية
            return 10000.0 
        # ضرب سعر النحاس في 2204 (تحويل للوحدة المطلوبة)
        return float (data ["Close"].iloc [-1 ])*2204 
    except Exception as e :
        print ("Error fetching Copper price from Yahoo Finance:",e )
        return 10000.0 


def get_market_sentiment_from_news ():
    try :
        url ="https://newsapi.org/v2/everything"
        params ={
        "q":"commodities OR copper OR metals OR economy",
        "language":"en",
        "sortBy":"publishedAt",
        "apiKey":"d13a4e5c597c4073ac9906f7bf274901"
        }
        response =requests .get (url ,params =params )
        data =response .json ()

        articles =[]
        for a in data .get ("articles",[]):
            title =a .get ("title")or ""
            desc =a .get ("description")or ""
            combined =(title +" "+desc ).strip ()
            if combined :
                articles .append (combined )

        positive_words =["growth","increase","strong","positive","recovery"]
        negative_words =["decline","drop","weak","negative","crisis"]

        score =0 
        for text in articles :
            text_lower =text .lower ()
            score +=sum (1 for w in positive_words if w in text_lower )
            score -=sum (1 for w in negative_words if w in text_lower )

        return score /10 
    except Exception as e :
        print ("NewsAPI error:",e )
        return 0.0 


@app .route ("/api_predict")
def auto_predict ():
    try :
        df =load_data_from_db ()
        df =clean_dataframe (df )

        oil_price =get_brent_price ()
        usd_index =get_dxy_value ()
        news_sentiment =get_market_sentiment_from_news ()
        copper_price =get_copper_price_from_yfinance ()

        other =df [[col for col in feature_columns if col not in ["oil_price","usd_index","market_sentiment","copper_price"]]].mean ().values .tolist ()

        full_features =[
        other [0 ],
        oil_price ,
        usd_index ,
        other [1 ],
        other [2 ],
        news_sentiment ,
        other [3 ],
        copper_price 
        ]

        scaled =scaler .transform ([full_features ])
        prediction =model .predict (scaled )[0 ]
        features_dict ={
        'global_demand_index':full_features [0 ],
        'oil_price':full_features [1 ],
        'usd_index':full_features [2 ],
        'china_industry_output':full_features [3 ],
        'energy_cost_index':full_features [4 ],
        'market_sentiment':full_features [5 ],
        'supply_disruption_index':full_features [6 ],
        'copper_price':full_features [7 ]
        }


        save_prediction_to_db (features_dict ,prediction )


        save_prediction_to_copper_data (features_dict ,prediction )


        return jsonify ({
        'predicted_copper_price':float (prediction ),
        'oil_price':features_dict ['oil_price'],
        'usd_index':features_dict ['usd_index'],
        'market_sentiment':features_dict ['market_sentiment'],
        'copper_price':features_dict ['copper_price'],
        'features_used':features_dict ,
        'datetime':datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e :
        print ('Error in auto_predict:',e )
        return jsonify ({'error':str (e )}),500 





@app .route ('/upload',methods =['GET','POST'])
def upload_csv ():
    if request .method =='GET':
        return render_template ('upload.html')


    try :
        file =request .files .get ('csv_file')
        if not file :
            return render_template ('upload.html',flash_message ='لم يتم إرسال ملف',flash_success =False )


        new_df =pd .read_csv (file )


        existing_df =load_data_from_db ()
        combined =pd .concat ([existing_df ,new_df ],ignore_index =True )
        combined =clean_dataframe (combined )


        if 'date'in combined .columns and not combined .empty :
            combined ['date_datetime']=pd .to_datetime (combined ['date'],errors ='coerce',infer_datetime_format =True )
            df_valid =combined [combined ['date_datetime'].notna ()].copy ()
            df_invalid =combined [combined ['date_datetime'].isna ()].copy ()

            if not df_valid .empty :
                df_valid =df_valid .sort_values ('date_datetime',ascending =False ,na_position ='last',kind ='mergesort')
                df_valid =df_valid .reset_index (drop =True )
                df_valid ['date']=df_valid ['date_datetime'].apply (lambda x :x .strftime ('%Y-%m-%d')if pd .notna (x )else None )

            if not df_invalid .empty :
                df_invalid =df_invalid .reset_index (drop =True )
                combined =pd .concat ([df_valid ,df_invalid ],ignore_index =True )
            else :
                combined =df_valid 

            if 'date_datetime'in combined .columns :
                combined =combined .drop (columns =['date_datetime'])

            combined =combined .drop_duplicates (subset =['date'],keep ='last')
            combined =combined [combined ['date'].notna ()]
            combined =combined .reset_index (drop =True )


        conn =sqlite3 .connect (DB_FILE )
        combined .to_sql ('copper_data',conn ,if_exists ='replace',index =False )
        conn .close ()

        stats ={
        'rows':int (combined .shape [0 ]),
        'columns':int (combined .shape [1 ]),
        'headers':combined .columns .tolist (),
        'columns_match':set (feature_columns ).issubset (combined .columns )
        }

        return render_template ('upload.html',flash_message ='تم رفع ودمج الملف بنجاح',flash_success =True ,stats =stats )
    except Exception as e :
        print ('Error in upload_csv:',e )
        return render_template ('upload.html',flash_message =f'حدث خطأ أثناء الرفع: {e }',flash_success =False )





@app .route ('/database')
def database_view ():
    return render_template ('database.html')


@app .route ('/database-data')
def database_data ():
    try :

        reorder_copper_data ()

        page =request .args .get ('page',1 ,type =int )
        per_page =request .args .get ('per_page',50 ,type =int )
        search =request .args .get ('search','',type =str )
        sort_column =request .args .get ('sort','date',type =str )
        sort_order =request .args .get ('order','desc',type =str )

        conn =sqlite3 .connect (DB_FILE )


        where_clause =""
        params =[]
        if search :
            where_clause ="WHERE date LIKE ? OR copper_price LIKE ? OR Next_Day_Copper_Price LIKE ?"
            search_pattern =f"%{search }%"
            params =[search_pattern ,search_pattern ,search_pattern ]


        conn .row_factory =sqlite3 .Row 
        cursor =conn .cursor ()
        cursor .execute ("PRAGMA table_info(copper_data)")
        columns =[row [1 ]for row in cursor .fetchall ()]

        if sort_column not in columns :
            sort_column ='date'


        if sort_column =='date':

            query_all =f"SELECT * FROM copper_data {where_clause }"
            cursor .execute (query_all ,params )
            rows_all =cursor .fetchall ()
            column_names =[description [0 ]for description in cursor .description ]
            df =pd .DataFrame (rows_all ,columns =column_names )


            if 'date'in df .columns and not df .empty :


                df ['date_datetime']=pd .to_datetime (df ['date'],errors ='coerce',infer_datetime_format =True )


                df_valid =df [df ['date_datetime'].notna ()].copy ()
                df_invalid =df [df ['date_datetime'].isna ()].copy ()


                if not df_valid .empty :
                    ascending_order =(sort_order .lower ()=='asc')


                    df_valid =df_valid .sort_values ('date_datetime',ascending =ascending_order ,na_position ='last',kind ='mergesort')
                    df_valid =df_valid .reset_index (drop =True )

                    df_valid ['date']=df_valid ['date_datetime'].apply (lambda x :x .strftime ('%Y-%m-%d')if pd .notna (x )else None )


                    if len (df_valid )>0 :
                        print (f"[database_data] الترتيب: {'تصاعدي'if ascending_order else 'تنازلي'}, أول تاريخ: {df_valid ['date'].iloc [0 ]}, آخر تاريخ: {df_valid ['date'].iloc [-1 ]}, عدد الصفوف: {len (df_valid )}")


                if not df_invalid .empty :
                    df_invalid =df_invalid .reset_index (drop =True )





                if not df_invalid .empty :
                    if ascending_order :

                        df =pd .concat ([df_valid ,df_invalid ],ignore_index =True )
                    else :

                        df =pd .concat ([df_valid ,df_invalid ],ignore_index =True )
                else :
                    df =df_valid 


                if 'date_datetime'in df .columns :
                    df =df .drop (columns =['date_datetime'])


            total =len (df )


            offset =(page -1 )*per_page 
            df =df .iloc [offset :offset +per_page ]
        else :

            order_by =f"ORDER BY {sort_column } {sort_order .upper ()}"


            count_query =f"SELECT COUNT(*) as total FROM copper_data {where_clause }"
            cursor .execute (count_query ,params )
            total =cursor .fetchone ()[0 ]


            offset =(page -1 )*per_page 
            query =f"SELECT * FROM copper_data {where_clause } {order_by } LIMIT ? OFFSET ?"
            params_with_limit =params +[per_page ,offset ]

            cursor .execute (query ,params_with_limit )
            rows =cursor .fetchall ()
            column_names =[description [0 ]for description in cursor .description ]
            df =pd .DataFrame (rows ,columns =column_names )


            if 'date'in df .columns and not df .empty :
                df ['date']=pd .to_datetime (df ['date'],errors ='coerce')
                df ['date']=df ['date'].apply (lambda x :x .strftime ('%Y-%m-%d')if pd .notna (x )else None )

        conn .close ()


        data =df .to_dict ('records')


        for row in data :
            for key ,value in row .items ():
                if key =='date':

                    if value is None :
                        row [key ]=None 
                    elif pd .isna (value ):
                        row [key ]=None 
                    elif isinstance (value ,str ):

                        if value .strip ()==''or value =='NaT'or value =='nan':
                            row [key ]=None 
                        else :
                            row [key ]=value 
                    else :

                        row [key ]=str (value )if value is not None else None 
                elif isinstance (value ,(int ,float ))and key !='date':
                    if pd .isna (value ):
                        row [key ]=None 
                    else :
                        row [key ]=round (float (value ),2 )if value is not None else None 
                elif pd .isna (value ):
                    row [key ]=None 

        return jsonify ({
        'success':True ,
        'data':data ,
        'pagination':{
        'page':page ,
        'per_page':per_page ,
        'total':total ,
        'pages':math .ceil (total /per_page )if per_page >0 else 0 
        }
        })
    except Exception as e :
        print ('Error in database_data:',e )
        import traceback 
        traceback .print_exc ()
        return jsonify ({'success':False ,'error':str (e )}),500 


@app .route ('/predictions-data')
def predictions_data ():
    try :
        page =request .args .get ('page',1 ,type =int )
        per_page =request .args .get ('per_page',50 ,type =int )
        search =request .args .get ('search','',type =str )
        sort_column =request .args .get ('sort','date',type =str )
        sort_order =request .args .get ('order','desc',type =str )

        conn =sqlite3 .connect (DB_FILE )
        conn .row_factory =sqlite3 .Row 
        cursor =conn .cursor ()


        where_clause =""
        params =[]
        if search :
            where_clause ="WHERE datetime LIKE ? OR date LIKE ? OR predicted_price LIKE ?"
            search_pattern =f"%{search }%"
            params =[search_pattern ,search_pattern ,search_pattern ]


        cursor .execute ("PRAGMA table_info(predictions)")
        columns =[row [1 ]for row in cursor .fetchall ()]

        if sort_column not in columns :
            sort_column ='date'


        if sort_column =='date':

            query_all =f"SELECT * FROM predictions {where_clause }"
            cursor .execute (query_all ,params )
            rows_all =cursor .fetchall ()
            column_names =[description [0 ]for description in cursor .description ]
            df =pd .DataFrame (rows_all ,columns =column_names )


            if 'date'in df .columns and not df .empty :

                df ['date']=pd .to_datetime (df ['date'],format ='%Y-%m-%d',errors ='coerce')

                if df ['date'].isna ().any ():
                    df ['date']=pd .to_datetime (df ['date'],errors ='coerce')


                df_valid =df [df ['date'].notna ()].copy ()
                df_invalid =df [df ['date'].isna ()].copy ()


                if not df_valid .empty :
                    ascending_order =(sort_order .lower ()=='asc')

                    df_valid =df_valid .sort_values ('date',ascending =ascending_order ,na_position ='last',kind ='mergesort')
                    df_valid =df_valid .reset_index (drop =True )


                if not df_invalid .empty :
                    df_invalid =df_invalid .reset_index (drop =True )
                    df =pd .concat ([df_valid ,df_invalid ],ignore_index =True )
                else :
                    df =df_valid 


                df ['date']=df ['date'].apply (lambda x :x .strftime ('%Y-%m-%d')if pd .notna (x )else None )


            total =len (df )


            offset =(page -1 )*per_page 
            df =df .iloc [offset :offset +per_page ]
        else :

            order_by =f"ORDER BY {sort_column } {sort_order .upper ()}"


            count_query =f"SELECT COUNT(*) as total FROM predictions {where_clause }"
            cursor .execute (count_query ,params )
            total =cursor .fetchone ()[0 ]


            offset =(page -1 )*per_page 
            query =f"SELECT * FROM predictions {where_clause } {order_by } LIMIT ? OFFSET ?"
            params_with_limit =params +[per_page ,offset ]

            cursor .execute (query ,params_with_limit )
            rows =cursor .fetchall ()
            column_names =[description [0 ]for description in cursor .description ]
            df =pd .DataFrame (rows ,columns =column_names )


            if 'date'in df .columns and not df .empty :
                df ['date']=pd .to_datetime (df ['date'],errors ='coerce')
                df ['date']=df ['date'].apply (lambda x :x .strftime ('%Y-%m-%d')if pd .notna (x )else None )

        conn .close ()


        data =df .to_dict ('records')


        for row in data :
            for key ,value in row .items ():
                if key =='date':

                    if pd .isna (value )or value =='NaT'or value =='nan'or value is None :
                        row [key ]=None 
                    elif isinstance (value ,str )and (value =='NaT'or value =='nan'):
                        row [key ]=None 
                elif isinstance (value ,(int ,float ))and key not in ['id','datetime','date','time','created_at']:
                    if pd .isna (value ):
                        row [key ]=None 
                    else :
                        row [key ]=round (float (value ),2 )if value is not None else None 
                elif pd .isna (value ):
                    row [key ]=None 

        return jsonify ({
        'success':True ,
        'data':data ,
        'pagination':{
        'page':page ,
        'per_page':per_page ,
        'total':total ,
        'pages':math .ceil (total /per_page )if per_page >0 else 0 
        }
        })
    except Exception as e :
        print ('Error in predictions_data:',e )
        import traceback 
        traceback .print_exc ()
        return jsonify ({'success':False ,'error':str (e )}),500 





@app .route ('/retrain',methods =['POST'])
def retrain ():
    try :
        import subprocess ,sys 
        script_path =Path (__file__ ).resolve ().parent /'model'/'train_model.py'
        result =subprocess .run ([sys .executable ,str (script_path )],capture_output =True ,text =True ,timeout =600 )
        out =result .stdout 
        err =result .stderr 
        if result .returncode !=0 :
            print ('Retrain stderr:',err )
            return render_template ('upload.html',flash_message =f'فشل إعادة التدريب: {err }',flash_success =False )

        return render_template ('upload.html',flash_message ='اكتملت إعادة تدريب النموذج بنجاح',flash_success =True )
    except Exception as e :
        print ('Error in retrain:',e )
        return render_template ('upload.html',flash_message =f'خطأ أثناء إعادة التدريب: {e }',flash_success =False )




init_predictions_table ()

if __name__ =='__main__':

    start_auto_update ()



    app .run (host ='0.0.0.0',port =5000 ,debug =True )
else :

    start_auto_update ()



