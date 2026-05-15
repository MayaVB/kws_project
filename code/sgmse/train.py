# train.py
import torch
import wandb
import argparse
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join

# Set CUDA architecture list and float32 matmul precision high
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging.")
          parser_.add_argument("--wandb_name", type=str, default=None, help="Name for wandb logger. If not set, a random name is generated.")
          parser_.add_argument("--ckpt", type=str, default=None, help="Resume training from checkpoint.")
          parser_.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
          parser_.add_argument("--save_ckpt_interval", type=int, default=50000, help="Save checkpoint interval.")
          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
     trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
     trainer_parser.add_argument("--devices", default="auto", help="How many gpus to use.")
     trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
     trainer_parser.add_argument("--max_epochs", type=int, default=-1, help="Number of epochs to train.")
     
     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

     # Initialize logger, trainer, model, datamodule
     model = ScoreModel(
          backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['ScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )

     # Set up logger configuration
     if args.nolog:
          logger = None
     else:
          logger = WandbLogger(project="sgmse", log_model=False, save_dir=args.log_dir, name=args.wandb_name)
          logger.experiment.log_code(".")

     # Set up callbacks for logger

     run_dir = join(args.log_dir, f'{str(logger.version)}-{args.wandb_name}') if logger else None

     if logger != None:
          callbacks = [ModelCheckpoint(dirpath=run_dir, save_last=True, filename='{epoch}-last')]
          callbacks += [ModelCheckpoint(dirpath=run_dir,
               filename='{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval)]
          if args.num_eval_files:
               # Save best checkpoints according to metrics
               # best PESQ checkpoint - perceptual quality
               checkpoint_callback_pesq = ModelCheckpoint(dirpath=run_dir, 
                    save_top_k=1, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
               # best ESTOI checkpoint - intelligibility
               checkpoint_callback_estoi = ModelCheckpoint(dirpath=run_dir,
                    save_top_k=1, monitor="estoi", mode="max", filename='{epoch}-{estoi:.2f}')
               callbacks += [checkpoint_callback_estoi, checkpoint_callback_pesq]
               # best SI-SDR checkpoint - signal reconstruction quality
               checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=run_dir, 
                    save_top_k=1, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
               # best SI-SIR checkpoint - interference suppression quality
               checkpoint_callback_si_sir = ModelCheckpoint(dirpath=run_dir,
                    save_top_k=1, monitor="si_sir", mode="max", filename='{epoch}-{si_sir:.2f}')
               # best SI-SAR checkpoint - artifact reduction quality
               checkpoint_callback_si_sar = ModelCheckpoint(dirpath=run_dir,
                    save_top_k=1, monitor="si_sar", mode="max", filename='{epoch}-{si_sar:.2f}')
               
               callbacks += [checkpoint_callback_si_sdr, checkpoint_callback_si_sir, checkpoint_callback_si_sar]
               
               # Save checkpoints every 50 epochs
               checkpoint_callback_epochs = ModelCheckpoint(
               dirpath=run_dir,
               filename='epoch{epoch}', save_top_k=-1, every_n_epochs=50)

               # Save checkpoint with best validation loss - for monitoring training progress
               #  and early stopping if necessary
               checkpoint_callback_valid_loss = ModelCheckpoint(
               dirpath=run_dir,
               save_top_k=1, monitor="valid_loss", mode="min", filename='{epoch}-{valid_loss:.4f}')

               callbacks += [checkpoint_callback_valid_loss, checkpoint_callback_epochs]
             
     else:
          callbacks = None

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer(
          **vars(arg_groups['Trainer']),
          strategy="ddp", 
          # strategy='auto',
          logger=logger,
          default_root_dir=args.log_dir,
          log_every_n_steps=10, num_sanity_val_steps=0,
          callbacks=callbacks
     )

     # Train model
     trainer.fit(model, ckpt_path=args.ckpt)