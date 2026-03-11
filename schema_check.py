import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
dsn = os.getenv('DATABASE_URL').split('?')[0]
conn = psycopg2.connect(dsn)
cur = conn.cursor()

tables = [
    'master_sale', 'invoice', 'invoice_details', 'orders', 'order_details',
    'ims_sale', 'ims_brick', 'products', 'product_groups', 'customers',
    'customer_details', 'doctors', 'doctor_plan', 'managers', 'healthcentres',
    'regions', 'zones', 'areas', 'territories', 'targets', 'roi'
]

print("--- TABLE STRATEGY ---")
for t in tables:
    try:
        cur.execute(f"SELECT COUNT(1) FROM {t}")
        count = cur.fetchone()[0]
        print(f"{t}: {count} rows")
    except Exception as e:
        conn.rollback()
        print(f"{t}: Error {e}")

conn.close()
