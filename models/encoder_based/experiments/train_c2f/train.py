import sys
sys.path.append('/home/mila/x/xiyuan.zou/research/kd-coref')
import logging
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from training_modules.c2f_model_no_distillation import TrainingModule_c2f_no_distillation
from evaluators.metrics import total_num_parameters

log = logging.getLogger(__name__)

@hydra.main(config_path='../../config', config_name='config_train_c2f')
def main(cfg : DictConfig) -> None:
    pl.seed_everything(cfg['seed'])
    
    # Initialize the training_module 
    training_module=TrainingModule_c2f_no_distillation(preprocessing_cfg=cfg["preprocessing_cfg"], model_cfg=cfg["model_cfg"], training_cfg=cfg["training_cfg"])
    print("model size (MB) is", total_num_parameters(training_module.model)/1000000)
    
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
                         devices=1,
                         val_check_interval=1.0,
                         **cfg['trainer'])
    trainer.fit(training_module, ckpt_path='/network/scratch/x/xiyuan.zou/kd-coref-project-output/lightning_logs/version_108/checkpoints/epoch=42-step=120486.ckpt')

if __name__ == '__main__':
    main()