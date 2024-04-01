"""Finetune an LM for coreference resolution using lightning."""
import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from transformers import AutoTokenizer

from data_modules.bert_coref_data_module import BertCorefDataModule
from lightning_modules.bert_coref_module import BertCorefModule
from data_modules.word_level_coref_data_module import WordLevelCorefDataModule
from lightning_modules.word_level_coref_module import WordLevelCorefModule
from data_modules.s2e_coref_data_module import S2ECorefDataModule
from lightning_modules.s2e_coref_module import S2ECorefModule

log = logging.getLogger(__name__)


@hydra.main(config_path='../config', config_name='config')
def main(cfg : DictConfig) -> None:
    """
    Finetune a model based on the given config.
    """
    pl.seed_everything(cfg['seed'])
    
    # Initialize model and data_module
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # TODO: move tokenizer name to config
    lightning_module = WordLevelCorefModule()
    data_module = WordLevelCorefDataModule(tokenizer=tokenizer, **cfg['data_module'])
    
    # Instantiate loggers
    loggers = []
    csv_logger = CSVLogger(save_dir=cfg['output_dir'])
    loggers.append(csv_logger)
    wandb_logger = WandbLogger(save_dir=cfg['output_dir'], project='llm-coref')
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
    trainer.fit(lightning_module, data_module, ckpt_path='last')


if __name__ == '__main__':
    main()
