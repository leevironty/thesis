from argparse import Namespace, ArgumentParser
import logging

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch

from lightning.pytorch.loggers.wandb import WandbLogger

from thesis.models.gnn.hgt import Predictor
from thesis.models.gnn.data import get_loaders


logger = logging.getLogger(__name__)


def attach_train_parser(main_subparsers):
    parser: ArgumentParser = main_subparsers.add_parser(name='train', help='Train GNN')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--train-share', type=float, default=0.8)
    parser.add_argument('--val-share', type=float, default=0.1)
    # parser.add_argument('--in_channels_node', type=int, default=15)
    # parser.add_argument('--edge_dim', type=int, default=15)
    # parser.add_argument('--out_channels_per_head', type=int, default=15)
    parser.add_argument('--num-layers', type=int, required=True)
    parser.add_argument('--num-heads', type=int, required=True)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--hidden-channels', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--accelerator', choices=['auto', 'cpu'], default='auto')
    # parser.add_argument('--run-eval', action=BooleanOptionalAction, default=False)
    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument(
        '--run-name', default='default-name', help='Name of the training run in WandB.'
    )
    parser.set_defaults(func=run_train)


def run_train(args: Namespace):
    logger.info(f'Started training with args: {args}')
    logger.info('Getting data loaders')
    train, val, test = get_loaders(
        args.dataset,
        threads=args.num_workers,
        # threads=1,  # multiple workers seems to slow down teardown??
        batch_size=args.batch_size,
        test_share=(1.0 - args.val_share - args.train_share),
        val_share=args.val_share,
    )
    logger.info('Setting up the model')
    first_sample = train.__iter__().__next__()
    # meta = first_sample.metadata()
    model = Predictor(
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        data=first_sample,
    )

    # initialize lazy parameters
    with torch.no_grad():
        model(first_sample)

    logger.info('Setting up the wandb logger')
    experiment_logger = WandbLogger(
        project='thesis',
        name=args.run_name,
        save_dir='wandb',
        log_model=False,
    )

    experiment_logger.watch(
        model,
        log_freq=500,
    )

    version = experiment_logger.version
    if version is None:
        version = 'no-version'

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{args.run_name}/{version}',
        filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss_epoch',
        mode='min',
        save_top_k=2,
    )

    lr_monitor_callback = LearningRateMonitor(
        logging_interval='step',
        log_momentum=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss_epoch',
        patience=args.patience,
        min_delta=0.0003,
    )

    # trainer = Trainer(
    #     default_root_dir='lightning_checkpoints',
    #     accelerator=args.accelerator,
    #     max_epochs=args.max_epochs,
    #     logger=experiment_logger,
    #     callbacks=[
    #         checkpoint_callback,
    #         early_stopping_callback,
    #     ],
    # )
    trainer = Trainer(
        default_root_dir='lightning_checkpoints',
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        logger=experiment_logger,
        accumulate_grad_batches=16,
        gradient_clip_val=0.5,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor_callback,
        ],
    )
    torch.set_float32_matmul_precision('medium')
    logger.info('Training')

    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)
    # trainer.fit(model=model, train_dataloaders=train)
    logger.info('All done')
