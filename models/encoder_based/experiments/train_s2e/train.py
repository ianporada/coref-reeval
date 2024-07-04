import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
import logging
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from training_modules.s2e_model_no_distillation import TrainingModule_s2e_no_distillation

log = logging.getLogger(__name__)

@hydra.main(config_path='../../config', config_name='config_train_s2e')
def main(cfg : DictConfig) -> None:
    pl.seed_everything(cfg['seed'])
    
    # Initialize the training_module 
    training_module=TrainingModule_s2e_no_distillation(preprocessing_cfg=cfg["preprocessing_cfg"], model_cfg=cfg["model_cfg"], training_cfg=cfg["training_cfg"])
    
    # Instantiate loggers
    loggers = []
    csv_logger = CSVLogger(save_dir=cfg['output_dir'])
    loggers.append(csv_logger)
    wandb_logger = WandbLogger(save_dir=cfg['output_dir'], project='KD-coref')
    wandb_logger.experiment.config.update(cfg)
    loggers.append(wandb_logger)
    
    # Instantiate callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(save_last=True)
    callbacks = [lr_monitor, checkpoint_callback]
    
    # Train
    # Loads additional keyword args from the config
    trainer = pl.Trainer(logger=loggers,
                         callbacks=callbacks,
                         accelerator='auto',
                         devices='auto',
                         val_check_interval=1.0,
                         **cfg['trainer'])
    trainer.fit(training_module, ckpt_path='last')

if __name__ == '__main__':
    main()