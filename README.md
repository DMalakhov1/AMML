# AMML — соревнования по прикладному машинному обучению

Репозиторий с решениями учебных соревнований по прикладному машинному обучению. В проектах собраны задачи из разных областей ML: оффлайн-обучение политик, динамическое ценообразование и ранжирование товаров в поиске.

Основной фокус репозитория — не просто обучение моделей, а полный ML-пайплайн: анализ данных, feature engineering, обучение моделей, валидация, постобработка и генерация `submission.csv` в требуемом формате.

## Проекты

| Проект | Задача | Метрика | Основные методы |
|---|---|---|---|
| [`competition_RL`](./competition_RL) | Offline RL / contextual bandits для e-mail marketing | SNIPS / IPS-based policy value | policy learning, per-arm modeling, uplift logic, stochastic policy |
| [`competition_pricing`](./competition_pricing) | Dynamic pricing: предсказание нижней и верхней границ цены | mean 1D IoU | LightGBM Quantile Regression, Gradient Boosting, Random Forest, QuantileRegressor, feature engineering |
| [`competition_ranking`](./competition_ranking) | Ранжирование товаров по поисковому запросу | nDCG@10 | BM25, TF-IDF, SVD, Word2Vec, sentence-transformers, LightGBM LambdaRank, CatBoostRanker, XGBoost Ranker |

## 1. Offline RL / Contextual Bandits

**Папка:** [`competition_RL`](./competition_RL)

### Задача

Нужно обучить политику выбора e-mail кампании. Для каждого пользователя модель получает контекст и возвращает распределение вероятностей по трём возможным действиям:

- `Mens E-Mail`;
- `Womens E-Mail`;
- `No E-Mail`.

Цель — максимизировать ожидаемую вероятность посещения сайта.

### Данные

Обучающая выборка содержит **48 000 строк** и **11 признаков**:

- контекст пользователя: `recency`, `history`, `mens`, `womens`, `newbie`, `zip_code`, `channel`, `history_segment`;
- действие логирующей политики: `segment`;
- целевое действие / reward: `visit`.

Целевая переменная несбалансирована: доля положительных посещений составляет примерно **14,7%**.

### Метод

Задача рассматривается как offline policy learning, а не как обычная классификация. Основная сложность в том, что для каждого пользователя наблюдается только одно действие логирующей политики, а результаты альтернативных действий неизвестны.

В решении используются:

- предобработка числовых и категориальных признаков;
- feature engineering на основе истории пользователя;
- оценка полезности действия для каждого варианта коммуникации;
- генерация стохастической политики в виде вероятностей по всем действиям;
- валидационная логика на основе IPS / SNIPS.

### Формат результата

Итоговый `submission.csv` имеет следующий вид:

```csv
id,p_mens_email,p_womens_email,p_no_email
100001,0.10,0.80,0.10
100002,0.33,0.33,0.34
```

## 2. Dynamic Pricing

**Папка:** [`competition_pricing`](./competition_pricing)

### Задача

Нужно предсказать справедливый ценовой интервал для каждого товара и даты. Для каждого объекта модель предсказывает два значения:

- `price_p05` — нижняя граница ценового интервала;
- `price_p95` — верхняя граница ценового интервала.

### Данные

В репозитории находятся подготовленные данные:

- `train.csv`: **29 100 строк**, 19 колонок;
- `test.csv`: **28 050 строк**, 18 колонок;
- `sample_submission.csv`: шаблон итогового файла.

Данные включают признаки товара, категории, календаря, погоды, праздников, активности и магазина.

### Метрика

Соревнование оценивается по **mean 1D Intersection over Union** между реальным и предсказанным ценовыми интервалами.

Модель должна не только предсказывать близкие значения, но и формировать корректный интервал:

```text
price_p05 < price_p95
```

### Метод

Решение использует ансамбль регрессионных моделей, обученных отдельно для `price_p05` и `price_p95`.

Основные элементы решения:

- календарные признаки: день, неделя, месяц, квартал, сезонные флаги;
- циклические признаки для дня недели, месяца, недели и квартала;
- погодные взаимодействия и полиномиальные преобразования;
- lag features и rolling statistics по `product_id`;
- expanding statistics и признаки тренда;
- frequency encoding и target encoding для категориальных переменных;
- кластеризация товаров и категорий с помощью `KMeans`;
- детекция аномалий через `IsolationForest`;
- снижение размерности погодных признаков через `PCA`;
- ансамбль из `LightGBM`, `GradientBoostingRegressor`, `RandomForestRegressor` и `QuantileRegressor`;
- оптимизация весов ансамбля напрямую под IoU-метрику;
- постобработка для гарантии корректного ценового интервала.

### Как запустить

```bash
cd competition_pricing
pip install -r requirements.txt
python main.py
```

После запуска скрипт создаёт файл:

```text
results/submission.csv
```

## 3. Product Search Ranking

**Папка:** [`competition_ranking`](./competition_ranking)

### Задача

Нужно ранжировать товары по поисковому запросу. Для каждого запроса модель получает список товаров-кандидатов и предсказывает релевантность каждой пары `запрос — товар`.

Цель — поднять наиболее релевантные товары выше в поисковой выдаче.

### Целевая переменная

Релевантность размечена по шкале от 0 до 3:

- `3` — высокая релевантность;
- `2` — релевантный товар;
- `1` — частичная релевантность;
- `0` — нерелевантный товар.

### Метрика

Соревнование оценивается по **nDCG@10**. Метрика поощряет модели, которые ставят наиболее релевантные товары в верхнюю часть выдачи.

### Метод

Решение объединяет классические IR-признаки, семантические признаки и learning-to-rank модели.

Feature engineering включает:

- очистку и нормализацию текста;
- исправление опечаток и повторяющихся чисел;
- BM25-признаки: Okapi BM25, BM25Plus, BM25L;
- TF-IDF cosine similarity;
- SVD-компоненты из TF-IDF векторов;
- similarity-признаки на основе Word2Vec;
- Levenshtein similarity;
- n-gram и character n-gram overlap;
- Jaccard, Dice и overlap coefficients;
- IDF-weighted overlap;
- position-based features;
- entity и noise features;
- статистики на уровне запроса;
- pairwise и rank-normalized признаки внутри каждой группы запросов;
- Reciprocal Rank Fusion features;
- опциональные семантические признаки из `sentence-transformers` и cross-encoder scoring.

Модели:

- `LightGBM` с LambdaRank objective;
- `CatBoostRanker` с YetiRank loss;
- `XGBoost Ranker`;
- финальный weighted ensemble с калибровкой на уровне запроса.

### Как запустить

```bash
cd competition_ranking
pip install -r requirements.txt
python main.py
```

После запуска скрипт создаёт файл:

```text
results/submission.csv
```

## Структура репозитория

```text
AMML/
├── competition_RL/
│   ├── Readme.md
│   ├── competition_1.ipynb
│   └── competition_1_AMML_DA.ipynb
│
├── competition_pricing/
│   ├── data/
│   ├── results/
│   ├── EDA_.ipynb
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
│
├── competition_ranking/
│   ├── data/
│   ├── results/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
└── README.md
```

## Результаты

Текущая версия репозитория содержит пайплайны решений и сгенерированные submission-файлы. Финальные значения leaderboard-метрик нужно добавить после проверки результатов.

| Проект | Local validation | Public score | Private score |
|---|---:|---:|---:|
| Offline RL / contextual bandits | TBD | TBD | TBD |
| Dynamic pricing | TBD | TBD | TBD |
| Product ranking | TBD | TBD | TBD |

## Стек

- Python
- pandas, NumPy
- scikit-learn
- LightGBM
- CatBoost
- XGBoost
- sentence-transformers
- rank-bm25
- gensim
- Docker

## Что показывает этот репозиторий

Репозиторий демонстрирует практические навыки, важные для ролей Data Scientist и Machine Learning Engineer:

- перевод условий соревнования в воспроизводимый ML-пайплайн;
- построение сильного feature engineering для табличных и текстовых данных;
- работа с ranking, pricing и policy learning задачами;
- выбор валидации с учётом бизнесовой метрики;
- объединение нескольких моделей в ансамбль;
- подготовка production-like `submission.csv` с Docker-compatible структурой.

## Планируемые улучшения

- Добавить финальные public/private leaderboard scores для каждого соревнования.
- Удалить или спрятать дублирующие template-папки, если они не используются в финальной версии.
- Добавить краткие EDA-выводы с ключевыми графиками по каждой задаче.
- Добавить отдельный `requirements.txt` для RL-ноутбука.
- Добавить validation reports с метриками и feature importance.
