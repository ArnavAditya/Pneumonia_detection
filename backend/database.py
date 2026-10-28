import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'predictions.db')


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            notes TEXT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            gradcam_image TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(patient_name, age, gender, notes, filename, prediction, confidence, gradcam_image):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO patient_history
        (patient_name, age, gender, notes, filename, prediction, confidence, gradcam_image, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_name,
        age,
        gender,
        notes,
        filename,
        prediction,
        confidence,
        gradcam_image,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, patient_name, age, gender, notes, filename, prediction, confidence, gradcam_image, created_at
        FROM patient_history
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def search_history_by_name(name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, patient_name, age, gender, notes, filename, prediction, confidence, gradcam_image, created_at
        FROM patient_history
        WHERE patient_name LIKE ?
        ORDER BY id DESC
    """, (f"%{name}%",))

    rows = cursor.fetchall()
    conn.close()
    return rows