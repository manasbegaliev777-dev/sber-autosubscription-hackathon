# Анализ сайта СберАвтоподписки — Хакатон

**Курс:** Машинное обучение 
**Тип работы:** Проектная практика (хакатон)

**GitHub:** https://github.com/manasbegaliev777-dev/sber-autosubscription-hackathon  
**Датасет (скачать):** https://cloud.mail.ru/public/PXoc/hDmWMRLe6

---

## О проекте

Компания «СберАвтоподписка» — сервис долгосрочной аренды автомобилей для физических лиц. В рамках проекта построена модель, которая предсказывает вероятность совершения целевого действия пользователем на сайте (оставить заявку, заказать звонок и др.).

Модель позволяет:
- оценивать эффективность каналов привлечения трафика
- адаптировать рекламные кампании под нужную аудиторию
- улучшать UX сайта на основе анализа поведения пользователей

---

## Структура репозитория

```
├── 01_data_loading.ipynb             # Загрузка данных и первичный анализ
├── 02_EDA.ipynb                      # Разведочный анализ данных (EDA)
├── 03_ML_model.ipynb                 # Построение и обучение ML-модели
├── 04_API.ipynb                      # Тестирование API
├── predict.py                        # FastAPI-сервис для предсказаний
├── model.pkl                         # Обученная модель CatBoost
├── features_meta.pkl                 # Метаданные признаков
├── sber_autoподписка_hackathon.pptx  # Презентация проекта
├── swagger_ui_testing.pdf            # Скриншот тестирования API через Swagger UI
├── requirements.txt                  # Зависимости проекта
└── README.md
```

---

## Данные

Данные из Google Analytics (last-click attribution) по сайту «СберАвтоподписки».

| Файл | Описание | Строк | Колонок |
|------|----------|-------|---------|
| `ga_sessions.pkl` | Один визит = одна строка. Содержит utm, device, geo и др. | 1 860 042 | 18 |
| `ga_hits.pkl` | Одно событие в рамках сессии | 15 726 470 | 11 |

> Датасеты не загружены в репозиторий из-за большого размера (суммарно ~5 ГБ).  
> Скачать данные можно по ссылке: https://cloud.mail.ru/public/PXoc/hDmWMRLe6

**Целевая переменная:** сессия считается конверсионной если в ней зафиксировано хотя бы одно из 7 целевых действий:
- `sub_car_claim_click` — 37 928 событий
- `sub_submit_success` — 18 439 событий
- `sub_car_claim_submit_click` — 12 359 событий
- `sub_call_number_click` — 3 653 события
- `sub_callback_submit_click` — 3 074 события
- `sub_car_request_submit_click` — 2 966 событий
- `sub_custom_question_submit_click` — 619 событий

**Конверсия (CR):** 2.10% (39 030 целевых сессий из 1 860 042)

---

## Ход работы

### 1. Загрузка и первичный анализ (`01_data_loading.ipynb`)

- Загрузка обоих датасетов через `pd.read_pickle()`
- Изучение структуры, типов данных и пропусков
- Определение целевых действий и формирование целевой переменной `target`
- Зафиксирован сильный дисбаланс классов: 97.90% / 2.10%
- Сохранён `sessions_with_target.pkl` для следующего этапа

### 2. Разведочный анализ (`02_EDA.ipynb`)

Построено 11 графиков с выводами по каждому.

**Ключевые находки:**

| Наблюдение | Факт |
|---|---|
| Тип устройства | Mobile — 72% трафика, но desktop конвертирует в 1.8× лучше |
| Лучший канал по CR | referral: **4.21%** |
| Органический трафик | CR 2.43%, прямые заходы (none): 2.90% |
| Глубина сессии | Средняя: **8.4 хита**, медиана: 3.0 |
| Повторные визиты | visit_number положительно коррелирует с конверсией |
| Корреляция is_organic | +0.049 с target |
| Корреляция is_mobile | −0.007 с target |

### 3. ML-модель (`03_ML_model.ipynb`)

**Разбивка данных (80/20, GroupShuffleSplit по client_id):**
- Train: 1 487 448 строк | позитивных: 2.09%
- Test: 372 594 строк | позитивных: 2.12%
- Пересечение клиентов между train и test: **0** (все визиты одного клиента строго в одной выборке)

**Feature Engineering — 23 признака:**

*Числовые (16):* `visit_number`, `log_visit_number`, `hour`, `weekday`, `month`, `is_organic`, `is_mobile`, `is_desktop`, `is_tablet`, `is_night`, `is_work_hours`, `is_weekend`, `is_first_visit`, `is_returning`, `is_ios`, `screen_width`

> `hits_count` и `log_hits_count` исключены: эти признаки вычисляются по всей сессии и захватывают события после целевого действия, что создаёт утечку данных (data leakage). Модель на их основе подглядывала бы в будущее.

*Категориальные (7):* `utm_medium`, `utm_source`, `device_category`, `device_os`, `device_browser`, `geo_city`, `geo_country`

**Параметры CatBoost:**
```python
CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,   # небольшой шаг для стабильного обучения
    depth=6,              # баланс между сложностью и переобучением
    scale_pos_weight=46.8, # компенсация дисбаланса классов 97.9/2.1
    eval_metric='AUC',
    random_seed=42
)
```

**Результаты:**

| Модель | ROC-AUC |
|--------|---------|
| DummyClassifier (random baseline) | 0.5006 |
| Logistic Regression (бейзлайн) | 0.6305 |
| **CatBoost (финальная модель)** | **0.7197** |
| Целевой порог по заданию | ≥ 0.6500 |

Целевой порог превышен на **+0.070**.

**Прогресс обучения:**
```
0:    AUC = 0.6630
100:  AUC = 0.7099
200:  AUC = 0.7152
300:  AUC = 0.7180
400:  AUC = 0.7207
499:  AUC = 0.7220  ← лучший результат
```

**Топ-10 признаков по важности (CatBoost Feature Importance):**

| Признак | Важность |
|---------|----------|
| utm_medium | 18.2% |
| utm_source | 15.7% |
| is_organic | 13.2% |
| screen_width | 11.5% |
| month | 9.8% |
| log_visit_number | 4.1% |
| device_category | 4.0% |
| device_os | 3.9% |
| device_browser | 3.3% |
| visit_number | 2.5% |

**SHAP-анализ (направление влияния):**
- `utm_medium (organic/referral)` → резко увеличивает вероятность конверсии
- `visit_number ↑` → повторные визиты положительно влияют
- `is_mobile` → снижает вероятность конверсии
- `utm_source` → органические источники повышают; banner снижает
- `hour (ночь)` → ночная аудитория конвертирует выше среднего

### 4. API (`predict.py` + `04_API.ipynb`)

Реализован REST API на FastAPI с четырьмя эндпоинтами:

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Описание сервиса |
| GET | `/health` | Проверка работоспособности |
| POST | `/predict` | Возвращает 0 или 1 |
| POST | `/predict_proba` | Возвращает вероятность 0.0–1.0 |

**Производительность (1000 запросов):**

| Метрика | Значение |
|---------|----------|
| Среднее время ответа | 5.27 мс |
| Медиана | 5.14 мс |
| 99-й перцентиль | 6.46 мс |
| Максимум | 8.77 мс |
| Критерий ≤ 3000 мс | **выполнен** |

---

## Запуск

### Установка зависимостей

**Через conda (рекомендуется):**

```bash
conda create -n sber python=3.10
conda activate sber
pip install -r requirements.txt
```

**Или напрямую через pip:**

```bash
pip install catboost lightgbm shap fastapi uvicorn joblib pandas numpy scikit-learn matplotlib seaborn
```

### Запуск API

```bash
cd "путь/к/папке/проекта"
uvicorn predict:app --reload --port 8000
```

После запуска:
- Сервис доступен: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "visit_number": 5,
       "hour": 14,
       "weekday": 2,
       "month": 3,
       "utm_medium": "organic",
       "utm_source": "google",
       "device_category": "desktop",
       "device_os": "Windows",
       "device_browser": "Chrome",
       "device_screen_resolution": "1920x1080",
       "geo_country": "Russia",
       "geo_city": "Moscow"
     }'
```

**Ответ:**
```json
{
  "prediction": 1,
  "label": "целевое действие",
  "response_time": "0.0088 сек"
}
```

---

## Критерии оценки и результаты

| Критерий | Макс. баллов | Результат |
|----------|-------------|-----------|
| EDA | 25 | Обработка пропусков, 11 графиков, корреляции, выводы |
| ML-модель | 35 | ROC-AUC = **0.7197** (цель ≥ 0.65), утечка данных устранена |
| Интерпретация фичей | 10 | Feature Importance + SHAP (bar + beeswarm) |
| API / скрипт | 15 | FastAPI, 4 эндпоинта, время ответа < 9 мс |
| Презентация | 15 | `.pptx` прилагается |
| **Итого** | **100** | |

---

## Технологии

![Python](https://img.shields.io/badge/Python-3.10-blue)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Pandas](https://img.shields.io/badge/Pandas-2.1-blue)
![SHAP](https://img.shields.io/badge/SHAP-0.44-orange)

- **Python 3.10** + Anaconda
- **CatBoost** — основная модель
- **Scikit-learn** — бейзлайн, метрики
- **SHAP** — интерпретация модели
- **FastAPI + Uvicorn** — REST API
- **Pandas, NumPy** — обработка данных
- **Matplotlib, Seaborn** — визуализация
