# ai_spines_segmentation


# 1 этап - бинаризация

## обучение
### подготовка данных
упаковка в тензоры исходных снимков
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage1.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage1/finetune_5im.json" --out_dir "C:/Users/Student/datasets/CVstage1/finetune" --mode train --workers 8`
 для валидации во время обучения
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage1.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage1/9009_d_disk.json" --out_dir "D:/shteinberg/9009/ai_segm" --mode val --workers 4`


### запуск обучения

 `C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/train.py --config-name=train_stage1`

### просмотр графиков функции потерь и метрик на валидации 
 `C:/Users/Student/.conda/envs/ai_segm/python.exe -m tensorboard.main --logdir ../../../datasets/CVstage1/tensorboard_logs_lora/`

## предсказание
### подготовка данных
1. при предсказании на вход подаются целые снимки, формирование признаков и нарезка происходит автоматически


### запуск прогноза
для пачки из конфига (используется для формирования датасета на обучение на след. этапе)
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/generate_to_stage2.py`



Этот скрипт нужен был для склефки нарезанных патчей обратно в снимок для визуального контроля. Сейчас он не нужен, потому что есть скрипт visualP_analyze.py
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage1_for_image.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage2/folds1.json" --out_dir "C:/Users/Student/datasets/CVstage2/statia" --target_name in_vitro_3 --preds_dir "C:/Users/Student/datasets/CVstage2/L_dendrite"`



# 2 этап - восстановление шей
## обучение
### подготовка данных
1. генерация прогноза 1 модели для подготовки входных данных 2 этапа
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/generate_to_stage2.py`

### Создание датасета
  `& C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage2.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage2/folds1.json" --out_dir "C:/Users/Student/datasets/CVstage2/folds1_1" --preds_dir "C:/Users/Student/datasets/CVstage2/L_dendrite_1" --mode train --workers 8`

`--json_path` - путь к конфигу, где все снимки для создания датасета
`--out_dir` - куда сохранится датасет
`--preds_dir` - куда сохранили карты вероятностей с первого этапа
`--mode` датасет для обучения или валидации

### запуск обучения
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/train.py --config-name=train_stage2`

## предсказание
1. при предсказании на вход подаются целые снимки, формирование признаков и нарезка и нарезка происходит автоматически

### запуск прогноза

для пачки из конфига (используется для формирования датасета на обучение на след. этапе)
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/generate_to_stage3.py`

# 3 этап - восстановление масштаба
## обучение
### подготовка данных
1. получение инференса второго этапа
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/generate_to_stage3.py`

2. упаковка данных в тензоры поплиточно трейн выборки конфика для обучения 3 этапа
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage3.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage3/folds5.json" --out_dir "C:/Users/Student/datasets/CVstage3/folds5" --l1_dir "C:/Users/Student/datasets/CVstage2/L_dendrite" --l2_dir "C:/Users/Student/datasets/CVstage3/necks" --mode train --workers 1`

### запуск обучения
 `C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/train.py --config-name=train_stage3`

### просмотр графиков функции потерь и метрик на валидации 
 `C:/Users/Student/.conda/envs/ai_segm/python.exe -m tensorboard.main --logdir="C:/Users/Student/datasets/CVstage3/tensorboard_logs/stage3_model"`


## предсказание
### подготовка данных
1. при предсказании на вход подаются целые снимки, формирование признаков и нарезка происходит автоматически

запуск со скелетом. почему? Для эксперимента, основной вариант БЕЗ скелета
 `C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage3_with_sceleton.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage3/folds1.json" --out_dir "C:/Users/Student/datasets/CVstage3/folds1_08skel" --l1_dir "C:/Users/Student/datasets/CVstage2/L_dendrite_2" --l2_dir "C:/Users/Student/datasets/CVstage3/necks_08skel" --mode val --workers 2`

### запуск прогноза

для пачки из конфига (используется для формирования датасета на обучение на след. этапе)
 `C:/Users/Student/.conda/envs/ai_segm/python.exe generate_to_stage4.py`
# 4 этап - 
## обучение
### подготовка данных
1. генерация прогноза третьей модели для подготовки входных данных четвертого этапа
 `C:/Users/Student/.conda/envs/ai_segm/python.exe generate_to_stage4.py`

2. вычисление подготовительных данных - скелета поверхности
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage4_sceleton.py`

3. вычисление подготовительных данных - vsot сегментация на ствол и шипики
 `C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage4_vsot.py` 

4. вычисление матриц расстояний до оси ствола и осей шипиков
 `C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage4_distance.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage4/folds1_labid_dist.json" --workers 2`
все данные с этих 4 скриптов сохраняются в файлы .tiff в папки с исходным снимком(как укажем в конфиге json_path)

`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage4.py --json_path "data_preprocessing/CVstage4/folds5.json" --out_dir "C:/Users/Student/datasets/CVstage4/folds5" --mode train --base_path "../../" --workers 4`

### запуск обучения

`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/train.py --config-name=train_stage4`
### просмотр графиков функции потерь и метрик на валидации 

 `C:/Users/Student/.conda/envs/ai_segm/python.exe -m tensorboard.main --logdir ../../../datasets/CVstage4/tensorboard_logs_lora`

## предсказание
### подготовка данных
Шаги 1.-4. при формировании данных для обучения. 
при предсказании на вход подаются целые снимки, нарезка происходит автоматически

### запуск прогноза
для пачки из конфига. генерирует финальную сегментацию с 2 каналами - шипы и дендритный ствол
`C:/Users/Student/.conda/envs/ai_segm/python.exe generate_to_final_2ch.py`

что-то интересненькое. Нужен для сбора двух каналов после предсказания deepD3, VSOT. Делал для картинок в отчет
`C:/Users/Student/.conda/envs/ai_segm/python.exe multimask.py`


# Получение метрик по моделям
## расчет метрик по результатам прогноза 4 этапа
`C:/Users/Student/.conda/envs/ai_segm/python.exe src/models/utils/calculate_metrics_stage4.py -input_file stage4.json -output_folder C:\Users\Student\datasets\CVstage4\metrics1`


## расчет метрик по результатам прогноза первых трех этапов. 
пример запуска для 1го этапа:
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/src/utils/calculate_metrics_stage123.py -input_file="src/utils/stage1_overleap+sigma.json" -output_folder="C:/Users/Student/datasets/CVstage1/metrics/sigma2overleap32"`

пример запуска для 3 этапа:
 `C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/src/utils/calculate_metrics_stage123.py -input_file src/utils/stage3_test.json -output_folder "C:/Users/Student/datasets/CVstage3/metrics"`

Скрипт, который берет все нарезанные патчи(то что получилось при формировании датасета на обучение и валидацию), и склеивает их обратно. Полезно для визуального анализа этапов предобработки
`C:/Users/Student/.conda/envs/ai_segm/python.exe visual_analyze.py`

________________
Нужен чтобы приводить маски и зоны интересов к масштабу 0.1 0.1. Используется для обучения первых двух этапов
`C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/scale_mask.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/description_stage1_labid1.json" --out_dir "C:/Users/Student/datasets/CVstage1/mask_scale"`





Может быть полезно:
## stage1
### Создание датасета
  `& C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage1.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage1/folds1.json" --out_dir "C:/Users/Student/datasets/CVstage1/folds1_1" --mode train --workers 8`

`--json_path` - путь к конфигу, где все снимки для создания датасета
`--out_dir` - куда сохранится датасет
`--mode` датасет для обучения или валидации
*Нужно запустить для каждого folds(5 шт) и train/val.* **Итого 10 раз**
### Запуск обучения
1. Поменять в конфиг файле train_stage1.yaml пути к сохранению весов, логов и пути к датасетам train и val
```
   paths:
  checkpoint_dir: "../../../datasets/CVstage1/checkpoints_1/folds1"
  log_dir: "../../../datasets/CVstage1/tensorboard_logs_1/"

  datamodule:
    _target_: src.datamodules.dendrite_datamodule.DendriteDataModule

    train_patches_dir: "C:/Users/Student/datasets/CVstage1/folds1_1/train"
    val_patches_dir: "C:/Users/Student/datasets/CVstage1/folds1_1/val"
```

   2. Запустить обучение (файл train.py) Проверить, что указан нужный файл конфиг @hydra.main(version_base="1.3", config_path="configs", config_name="**train_stage1**")
`  & C:/Users/Student/.conda/envs/ai_segm/python.exe train.py`
3. Посмотреть логи: выполнить в консоли 
`   & C:/Users/Student/.conda/envs/ai_segm/python.exe -m tensorboard.main --logdir ../../../datasets/CVstage1/tensorboard_logs_1/ `

### Предсказать снимки
в файле **generate_to_stage2** поменять пути к конфигу, папки куда сохранится предсказание и бинаризация. Выбрать нужные веса для предсказания **ckpt** 
```
    JSON_PATH = "data_preprocessing/CVstage1/folds1.json"
    BASE_PATH = "../../" # базовый путь к картинкам
    OUT_DIR = "C:/Users/Student/datasets/CVstage2/L_dendrite_1/"
    OUT_DIR_bin = "C:/Users/Student/datasets/CVstage1/binarization_1/"
    
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_bin, exist_ok=True)

    ckpt = "../../../datasets/CVstage1/checkpoints_1/folds1/epoch=10-val_loss=0.0058.ckpt"
```

## stage2
### Создание датасета
  `& C:/Users/Student/.conda/envs/ai_segm/python.exe c:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/stage2.py --json_path "C:/Users/Student/source/repos/ai_spines_segmentation/data_preprocessing/CVstage2/folds1.json" --out_dir "C:/Users/Student/datasets/CVstage2/folds1_1" --preds_dir "C:/Users/Student/datasets/CVstage2/L_dendrite_1" --mode train --workers 8`

`--json_path` - путь к конфигу, где все снимки для создания датасета
`--out_dir` - куда сохранится датасет
`--preds_dir` - куда сохранили карты вероятностей с первого этапа
`--mode` датасет для обучения или валидации
*Нужно запустить для каждого folds(5 шт) и train/val.* **Итого 10 раз**

### Запуск обучения
1. Поменять в конфиг файле train_stage2.yaml пути к сохранению весов, логов и пути к датасетам train и val

```
   paths:
  checkpoint_dir: "../../../datasets/CVstage2/checkpoints_1/folds5"
  log_dir: "../../../datasets/CVstage2/tensorboard_logs_1/"

datamodule:
  _target_: src.datamodules.stage2_datamodule.Stage2DataModule
  # ПУТИ К ПАПКАМ, КОТОРЫЕ СОЗДАЛ СКРИПТ stage2.py
  train_patches_dir: "C:/Users/Student/datasets/CVstage2/folds1_1/train"
  val_patches_dir: "C:/Users/Student/datasets/CVstage2/folds1_1/val"
```
   2. Запустить обучение (файл train.py) Проверить, что указан нужный файл конфиг @hydra.main(version_base="1.3", config_path="configs", config_name="**train_stage2**")
`  & C:/Users/Student/.conda/envs/ai_segm/python.exe train.py`
3. Посмотреть логи: выполнить в консоли 
`   & C:/Users/Student/.conda/envs/ai_segm/python.exe -m tensorboard.main --logdir ../../../datasets/CVstage2/tensorboard_logs_2/ `


### Предсказать снимки
в файле **generate_to_stage3** поменять пути к конфигу и предсказаниям от 1 этапа, папки куда сохранится предсказание и бинаризация. Выбрать нужные веса для предсказания **ckpt** 
```
    JSON_PATH = "data_preprocessing/CVstage2/folds1.json"
    L1_DIR = "C:/Users/Student/datasets/CVstage2/L_dendrite_1"
    BASE_PATH = "../../" # базовый путь к картинкам
    OUT_DIR = "C:/Users/Student/datasets/CVstage3/necks_1"
    OUT_DIR_bin = "C:/Users/Student/datasets/CVstage2/bin_neck_1"

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_bin, exist_ok=True)

    ckpt = "../../../datasets/CVstage2/checkpoints/folds3/epoch=25-val_loss=0.1058.ckpt"
```