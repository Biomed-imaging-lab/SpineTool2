# import os
# отключаем строгий режим загрузки весов PyTorch 2.6+, чтобы можно было продолжить обучение загруженных весов
# os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(version_base="1.3", config_path="configs", config_name="train_stage1_lora")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    # 1. Init DataModule
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # 2. Init Model
    model = hydra.utils.instantiate(cfg.model)

    checkpoint = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir, 
        monitor="val/loss", 
        mode="min", 
        save_top_k=2,#сколько лучших эпох сохранить
        save_last=True,#сохранять ли последнюю эпоху
        filename="epoch={epoch:02d}-val_loss={val/loss:.4f}",
        auto_insert_metric_name=False
    )
    
    checkpoint_last = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        save_last=True,
        save_top_k=0,
    )
    logger = TensorBoardLogger(cfg.paths.log_dir, 
        name="folds",
        version=f"fold_{cfg.paths.fold}")
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # 3. Init Trainer
    trainer = pl.Trainer(
        logger=logger, callbacks=[checkpoint, checkpoint_last, lr_monitor],
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        # devices=cfg.trainer.devices,#stage3
        # accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,#stage3
        # precision=cfg.trainer.precision#stage3
    )
    # 4. Train
    trainer.fit(model=model, datamodule=datamodule)
    #если нужно продолжить обучение с какой то эпохи
    # trainer.fit(model=model, datamodule=datamodule, ckpt_path="C:/Users/Student/datasets/CVstage4/checkpoints/f_s2_08s/folds2/last-v1.ckpt")

if __name__ == "__main__":
    main()