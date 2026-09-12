import hydra
import torch
import pytorch_lightning as pl
from src.models.segmentation_module import DendriteSegmentationModule
from src.datamodules.dendrite_datamodule import DendriteDataModule

@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg):
    pl.seed_everything(42)

    # 1. Загружаем данные (Test Set)
    # Нам нужно, чтобы DataModule загрузил данные из папки test
    # В конфиге у нас нет ключа test_patches_dir, но мы можем передать его вручную или использовать val_ds логику
    
    # Самый простой способ: инициализируем модуль и подменяем путь валидации на тест
    # (так как логика валидации и теста одинакова - просто прогон и метрики)
    
    test_path = "C:/Users/Student/datasets/dataset_ai_spines_npz/test" # Путь к вашей папке с тестами
    
    print(f"Testing on data from: {test_path}")
    
    dm = DendriteDataModule(
        train_patches_dir=cfg.datamodule.train_patches_dir, # Не используется, но нужно для инит
        val_patches_dir=test_path,                          # <-- ПОДМЕНА: Грузим TEST как VAL
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers
    )
    
    # 2. Загружаем модель из чекпоинта
    # Укажите имя вашего лучшего чекпоинта!
    ckpt_path = "checkpoints/epoch=18-step=11153.ckpt" # Или конкретный файл, например "checkpoints/epoch_49.ckpt"
    
    try:
        model = DendriteSegmentationModule.load_from_checkpoint(ckpt_path, weights_only=False)
        print(f"Loaded weights from {ckpt_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 3. Запускаем тесты
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1,
        logger=False # Логи не нужны, вывод будет в консоль
    )
    
    # Используем validate, так как мы подменили папку. 
    # Это даст нам метрики val_dice и val_iou, посчитанные на тестовом наборе.
    trainer.validate(model, dm)

if __name__ == "__main__":
    main()