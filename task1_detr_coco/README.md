# Fine-tuning DETR on COCO2017 subset

### Цель

Освоить трансформерные архитектуры для объектной детекции, научиться настраивать и обучать DETR/Deformable-DETR на подмножестве COCO, анализировать loss и ошибки.

## Структура проекта

```text
.
├── notebooks/
│   ├── detr_coco_subset_finetuning.ipynb        # основной ноутбук для запуска эксперимента
│   └── artifacts/
│       └── runs/
│           └── detr_coco_subset_YYYYMMDD_HHMMSS/
│               ├── tensorboard/                 # логи TensorBoard
│               ├── profiler/                    # trace профайлера
│               ├── checkpoints/                 # сохранённые чекпойнты модели
│               │   ├── epoch_001/
│               │   ├── epoch_002/
│               │   └── epoch_003/
│               ├── plots/                       # графики loss и метрик
│               ├── visualizations/              # визуализации предсказанных боксов
│               ├── metrics_table.csv            # таблица метрик mAP и mAP50
│               ├── error_analysis.csv           # таблица error analysis
│               └── error_counts.csv             # сводка типов ошибок
├── requirements.txt                             # зависимости проекта
├── .gitignore
└── README.md
```

## Датасет

Данные взяты из сайта `kaggle`:
https://www.kaggle.com/datasets/abdelrahmanelgharibx/coco2017-subset

```python
import kagglehub
path = kagglehub.dataset_download("abdelrahmanelgharibx/coco2017-subset")
print("Path to dataset files:", path)
```

Структура данных:

```text
.../versions/1/
├── train2017.json
├── val2017.json
└── coco/
    ├── 000000217226.jpg
    ├── 000000393223.jpg
    └── ...
```


## Модель

В ноутбуке используется `facebook/detr-resnet-50` из Hugging Face Transformers. Классификационная голова заменяется под выбранные классы датасета с `ignore_mismatched_sizes=True`.

По умолчанию выбираются `10` самых частых классов из `train2017.json`.

## Основные гиперпараметры

```python
MODEL_NAME = "facebook/detr-resnet-50"
NUM_CLASSES = 10
EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-5
LR_BACKBONE = 1e-6
WEIGHT_DECAY = 1e-4
CONF_THRESHOLD_FOR_MAP = 0.01
CONF_THRESHOLD_FOR_VIS = 0.30
USE_PROFILER = True
```

## Как запустить

1. Создать окружение:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
```

2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Открыть ноутбук:

```bash
jupyter notebook notebooks/detr_coco_subset_finetuning.ipynb
```

4. Выполнить все ячейки сверху вниз.


## TensorBoard

Для просмотра loss, `mAP`, `mAP50` и trace профайлера:

```bash
tensorboard --logdir artifacts/runs
```

```text
http://localhost:6006
```


## Основные наблюдения

Для анализа ошибок были рассмотрены предсказания модели на валидационной выборке

### Типы ошибок в error analysis

| Тип ошибки | Пояснение |
|---|---|
| `correct` | Модель нашла объект правильно: предсказанный bounding box достаточно хорошо совпадает с ground truth box, и класс объекта определён верно. |
| `false_positive` | Модель предсказала лишний объект, которому не соответствует ни один реальный объект на изображении. Например, модель нарисовала bbox там, где объекта нужного класса нет. |
| `missed_object` | Реальный объект присутствует на изображении, но модель не смогла его обнаружить. То есть для ground truth объекта не нашлось подходящего предсказанного bbox. |
| `localization_error` | Класс объекта определён правильно, но bounding box расположен неточно: модель нашла нужный объект, но плохо оценила его границы. |
| `classification_error` | Bounding box расположен достаточно хорошо, но класс объекта определён неверно. То есть модель нашла объект, но перепутала его категорию. |
| `classification_and_localization_error` | Ошибка одновременно и в классе, и в локализации: модель предсказала неверный класс и при этом bbox недостаточно хорошо совпадает с реальным объектом. |

| Тип ошибки | Количество | Доля |
|---|---:|---:|
| False positive | 586 | 48.2% |
| Correct | 439 | 36.1% |
| Localization error | 87 | 7.2% |
| Missed object | 71 | 5.8% |
| Classification + localization error | 23 | 1.9% |
| Classification error | 10 | 0.8% |

Анализ ошибок показал, что основной тип ошибки — false positive: модель часто предсказывает лишние объекты. При этом чистых ошибок классификации мало, следовательно, модель в целом различает выбранные классы, но хуже справляется с фильтрацией лишних предсказаний и точной локализацией bounding boxes. Ошибки локализации и пропущенные объекты также присутствуют, но встречаются заметно реже, чем false positive.

| Epoch | Time, sec | LR | Train CE | Train BBox | Train GIoU | Train Card. Err. | Train Total Loss | Val CE | Val BBox | Val GIoU | Val Card. Err. | Val Total Loss | mAP | mAP50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 581.16 | 9.14e-06 | 0.7565 | 0.0399 | 0.2421 | 6.84 | 1.4401 | 0.5193 | 0.0488 | 0.2600 | 3.56 | 1.2831 | 0.0647 | 0.1168 |
| 2 | 571.91 | 6.89e-06 | 0.3949 | 0.0393 | 0.2376 | 3.88 | 1.0666 | 0.4142 | 0.0488 | 0.2616 | 4.27 | 1.1813 | 0.1629 | 0.2969 |
| 3 | 615.58 | 4.11e-06 | 0.3049 | 0.0369 | 0.2313 | 4.17 | 0.9521 | 0.3689 | 0.0468 | 0.2562 | 4.41 | 1.1152 | 0.1920 | 0.3620 |
| 4 | 575.69 | 1.86e-06 | 0.2622 | 0.0350 | 0.2252 | 4.48 | 0.8873 | 0.3532 | 0.0463 | 0.2589 | 4.18 | 1.1025 | 0.1993 | 0.3779 |
| 5 | 570.05 | 1.00e-06 | 0.2434 | 0.0335 | 0.2199 | 4.45 | 0.8507 | 0.3414 | 0.0464 | 0.2558 | 4.29 | 1.0850 | 0.2076 | 0.3878 |


В ходе fine-tuning DETR на COCO-subset наблюдалось стабильное снижение функции потерь и рост метрик детекции. Общий train loss уменьшился с `1.44` до `0.85`, а validation loss — с `1.28` до `1.08`, что говорит о корректном процессе обучения без явных признаков переобучения.

Наиболее заметное снижение произошло для classification loss: `train_loss_ce` уменьшился с `0.756` до `0.243`, а `val_loss_ce` — с `0.519` до `0.341`. Это показывает, что модель стала лучше различать выбранные 10 классов. Потери, связанные с локализацией (`loss_bbox`, `loss_giou`), также снижались, но менее резко, что указывает на то, что точная локализация объектов остаётся более сложной частью задачи.

Метрики качества детекции стабильно росли на протяжении всех эпох: `mAP` увеличился с `0.0647` до `0.2076`, а `mAP50` — с `0.1168` до `0.3878`.

