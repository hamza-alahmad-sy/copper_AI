"""
Script to migrate data from CSV to SQLite database
"""
import pandas as pd
import sqlite3
from pathlib import Path
import sys

# Set UTF-8 encoding for output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# مسارات الملفات
CSV_FILE = Path(__file__).resolve().parent / 'data' / 'copper_prediction_dataset_1000.csv'
DB_FILE = Path(__file__).resolve().parent / 'data' / 'copper_data.db'

def migrate_csv_to_sqlite():
    """تحويل البيانات من CSV إلى SQLite"""
    print("Reading CSV file...")
    df = pd.read_csv(CSV_FILE)
    
    print(f"Read {len(df)} rows of data")
    
    # إنشاء اتصال بقاعدة البيانات
    conn = sqlite3.connect(DB_FILE)
    
    # حفظ البيانات في جدول
    df.to_sql('copper_data', conn, if_exists='replace', index=False)
    
    # إنشاء فهرس على عمود التاريخ لتحسين الأداء
    conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON copper_data(date)')
    
    conn.commit()
    conn.close()
    
    print(f"Data saved to database: {DB_FILE}")
    print(f"Number of rows saved: {len(df)}")
    print("Migration completed successfully!")

if __name__ == '__main__':
    migrate_csv_to_sqlite()

