"""
predict.py — API для предсказания целевого действия пользователя на сайте СберАвтоподписки.

Принимает данные по визиту (utm, geo, device и др.), возвращает 0 или 1.

Запуск:
    uvicorn predict:app --reload --port 8000

Пример запроса:
    POST http://localhost:8000/predict
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import time

# ── Загрузка модели и метаданных ──────────────────────────────────────────────
MODEL_PATH    = r'D:\sber-autosubscription-hackathon\model.pkl'
FEATURES_PATH = r'D:\sber-autosubscription-hackathon\features_meta.pkl'

model         = joblib.load(MODEL_PATH)
features_meta = joblib.load(FEATURES_PATH)

ALL_FEATURES        = features_meta['all_features']
NUM_FEATURES        = features_meta['num_features']
CAT_FEATURES        = features_meta['cat_features']
CAT_FEATURE_INDICES = features_meta['cat_feature_indices']

app = FastAPI(
    title="СберАвтоподписка — Предсказание целевого действия",
    description="Модель предсказывает, совершит ли пользователь целевое действие на сайте.",
    version="1.0.0"
)


# ── Схема входных данных ───────────────────────────────────────────────────────
class VisitData(BaseModel):
    # Характеристики визита
    visit_number:             Optional[int]   = 1
    hour:                     Optional[int]   = 12
    weekday:                  Optional[int]   = 0
    month:                    Optional[int]   = 6

    # Источник трафика
    utm_medium:               Optional[str]   = 'unknown'
    utm_source:               Optional[str]   = 'unknown'

    # Устройство
    device_category:          Optional[str]   = 'mobile'
    device_os:                Optional[str]   = 'Android'
    device_browser:           Optional[str]   = 'Chrome'
    device_screen_resolution: Optional[str]   = '360x720'

    # География
    geo_country:              Optional[str]   = 'Russia'
    geo_city:                 Optional[str]   = 'Moscow'


# ── Вспомогательная функция: подготовка признаков ────────────────────────────
def prepare_features(data: VisitData) -> pd.DataFrame:
    visit_number = int(data.visit_number or 1)
    hour         = int(data.hour or 12)
    weekday      = int(data.weekday or 0)
    month        = int(data.month or 6)

    # Разрешение экрана — ширина
    try:
        screen_width = int(str(data.device_screen_resolution or '0').split('x')[0])
    except Exception:
        screen_width = 0

    utm_medium = str(data.utm_medium or 'unknown')
    device_os  = str(data.device_os  or 'unknown')

    row = {
        # Числовые
        'visit_number':    visit_number,
        'log_visit_number': np.log1p(visit_number),
        'hour':            hour,
        'weekday':         weekday,
        'month':           month,
        'is_organic':      int(utm_medium in ['organic', 'referral', '(none)']),
        'is_mobile':       int(str(data.device_category or '') == 'mobile'),
        'is_desktop':      int(str(data.device_category or '') == 'desktop'),
        'is_tablet':       int(str(data.device_category or '') == 'tablet'),
        'is_night':        int(hour in range(0, 7)),
        'is_work_hours':   int(hour in range(10, 20)),
        'is_weekend':      int(weekday >= 5),
        'is_first_visit':  int(visit_number == 1),
        'is_returning':    int(visit_number > 3),
        'is_ios':          int(device_os in ['iOS', 'Macintosh']),
        'screen_width':    screen_width,
        # Категориальные
        'utm_medium':      utm_medium,
        'utm_source':      str(data.utm_source or 'unknown'),
        'device_category': str(data.device_category or 'unknown'),
        'device_os':       device_os,
        'device_browser':  str(data.device_browser or 'unknown'),
        'geo_city':        str(data.geo_city    or 'unknown'),
        'geo_country':     str(data.geo_country or 'unknown'),
    }

    df = pd.DataFrame([row])[ALL_FEATURES]
    for col in CAT_FEATURES:
        df[col] = df[col].astype(str)
    for col in NUM_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


# ── Эндпоинты ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "СберАвтоподписка — предсказание целевого действия",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Предсказание: 0 или 1",
            "POST /predict_proba": "Вероятность целевого действия (0.0 — 1.0)",
            "GET  /health": "Проверка работоспособности сервиса"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(visit: VisitData):
    """
    Принимает данные по визиту, возвращает предсказание: 0 (нет действия) или 1 (целевое действие).
    """
    start = time.time()
    df    = prepare_features(visit)
    pred  = int(model.predict(df)[0])
    elapsed = round(time.time() - start, 4)

    return {
        "prediction":    pred,
        "label":         "целевое действие" if pred == 1 else "нет целевого действия",
        "response_time": f"{elapsed} сек"
    }


@app.post("/predict_proba")
def predict_proba(visit: VisitData):
    """
    Принимает данные по визиту, возвращает вероятность совершения целевого действия.
    """
    start = time.time()
    df    = prepare_features(visit)
    proba = float(model.predict_proba(df)[0][1])
    pred  = int(proba >= 0.5)
    elapsed = round(time.time() - start, 4)

    return {
        "probability":   round(proba, 4),
        "prediction":    pred,
        "label":         "целевое действие" if pred == 1 else "нет целевого действия",
        "response_time": f"{elapsed} сек"
    }
