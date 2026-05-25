# Домашнее задание 2.5: Synthetic Data with Stable Diffusion + ControlNet

### Цель

Научиться использовать генеративные модели для аугментации данных, исследовать влияние синтетических примеров на качество модели.

### Задание

1. Выбрать «редкие» классы из своего датасета.
2. Сгенерировать дополнительные изображения этих классов с помощью Stable Diffusion + ControlNet.
3. Добавить синтетику в тренировочный датасет.
4. Провести обучение базовой модели (CNN или ViT) без синтетики и с синтетикой.
5. Сравнить метрики и построить таблицу результатов (ablation).

## Структура проекта

```text
.
├── src/
│   ├── coco_utils.py
│   ├── prepare_crops.py
│   ├── generate_controlnet.py
│   ├── train_classifier.py
│   └── make_grids.py
├── artifacts/
│   ├── synthetic_examples/
│   ├── metrics/
│   ├── plots/
│   └── tensorboard/
├── data/
│   ├── crops/                  # реальные crop-ы объектов, генерируется скриптом
│   ├── synthetic_controlnet/    # синтетические изображения, генерируется скриптом
│   └── metadata/                # selected classes, metadata CSV
├── artifacts_rare_run/. 
│   ├── synthetic_examples/
│   ├── metrics/
│   ├── plots/
│   └── tensorboard/
├── data_rare_run/
│   ├── crops/                  
│   ├── synthetic_controlnet/    
│   └── metadata/ 
├── requirements.txt
├── run_on_vm.sh
└── README.md
```

### Датасет

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


## Как запустить на ВМ

### 1. Создать окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Установить зависимости

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

GPU:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
PY
```

### 3. Запустить pipeline

```bash
bash run_on_vm.sh
```

## Основные наблюдения

Для оценки влияния синтетических данных было проведено два эксперимента:

1. **Very rare classes** — классы с очень малым числом объектов валидации.
2. **Moderately rare classes** — более устойчивый вариант с большим числом train/val примеров.

В каждом эксперименте сравнивались две модели:

- `baseline_real_only` — ResNet18, обученный только на реальных crop-ах объектов из COCO.
- `real_plus_synthetic` — ResNet18, обученный на реальных crop-ах и дополнительных синтетических изображениях, сгенерированных через Stable Diffusion + ControlNet.

Validation set во всех экспериментах состоял только из реальных изображений. Синтетические изображения добавлялись только в train set.

---

### Experiment 1: very rare classes

В первом запуске были выбраны самые редкие классы из доступного COCO-subset.

| Class | Train objects | Val objects | Train crops |
|---|---:|---:|---:|
| hair drier | 93 | 5 | 93 |
| zebra | 347 | 11 | 250 |
| mouse | 369 | 19 | 250 |
| scissors | 431 | 11 | 250 |

Итого:

| Dataset | Train samples | Val samples |
|---|---:|---:|
| Real only | 843 | 46 |
| Real + synthetic | 1003 | 46 |

Результаты:

| Experiment | Accuracy | Macro F1 | Weighted F1 | Comment |
|---|---:|---:|---:|---|
| Real only | 0.978 | 0.982 | 0.978 | Best result |
| Real + synthetic | 0.935 | 0.913 | 0.934 | Worse than baseline |

В этом запуске добавление синтетических изображений не улучшило качество. Напротив, метрики снизились: accuracy уменьшилась с `0.978` до `0.935`, macro F1 — с `0.982` до `0.913`.

Основная причина — слишком маленький validation set: всего 46 изображений, причём для класса `hair drier` было только 5 validation-примеров. При таком размере выборки метрики становятся нестабильными: несколько ошибок могут сильно изменить accuracy и F1. Кроме того, baseline уже показал почти идеальное качество, поэтому синтетике было трудно дать дополнительный прирост.

Также возможен эффект domain gap: синтетические изображения, даже если они визуально похожи на нужный класс, отличаются от реальных COCO crop-ов по стилю, фону, текстурам и распределению объектов.

---

### Experiment 2: moderately rare classes

Во втором запуске были выбраны не самые редкие, а умеренно редкие классы. Для этого был увеличен нижний порог по количеству train objects.

| Class | Train objects | Val objects | Train crops |
|---|---:|---:|---:|
| microwave | 510 | 14 | 300 |
| toothbrush | 519 | 25 | 300 |
| toilet | 546 | 27 | 300 |
| parking meter | 596 | 33 | 300 |

Итого:

| Dataset | Train samples | Val samples |
|---|---:|---:|
| Real only | 1200 | 99 |
| Real + synthetic | 1360 | 99 |

Результаты:

| Experiment | Best accuracy | Best macro F1 | Best weighted F1 | Best val loss |
|---|---:|---:|---:|---:|
| Real only | 0.929 | 0.917 | 0.929 | 0.224 |
| Real + synthetic | 0.929 | 0.927 | 0.929 | 0.111 |

Во втором запуске синтетические данные дали небольшой положительный эффект. Accuracy осталась примерно на том же уровне, но macro F1 вырос с `0.917` до `0.927`, а validation loss снизился с `0.224` до `0.111`.

Это говорит о том, что при более устойчивом выборе классов синтетика может быть полезна: она увеличивает разнообразие train set и немного улучшает обобщающую способность модели. Особенно важен рост macro F1, потому что эта метрика одинаково учитывает все классы и лучше отражает качество на несбалансированных данных.

---

### Comparison of both runs

| Class selection | Experiment | Train samples | Val samples | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Very rare classes | Real only | 843 | 46 | 0.978 | 0.982 | 0.978 |
| Very rare classes | Real + synthetic | 1003 | 46 | 0.935 | 0.913 | 0.934 |
| Moderately rare classes | Real only | 1200 | 99 | 0.929 | 0.917 | 0.929 |
| Moderately rare classes | Real + synthetic | 1360 | 99 | 0.929 | 0.927 | 0.929 |
