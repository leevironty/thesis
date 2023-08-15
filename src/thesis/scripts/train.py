import thesis.models.gnn as gnn
from lightning import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from argparse import Namespace


def run_train(args: Namespace):
    model = gnn.demo.Demo()
    experiment_logger = WandbLogger(project='thesis')
    train, val, test = gnn.data.get_loaders(
        'solutions/data/toy2',
        threads=args.threads,
        batch_size=16,
    )
    trainer = Trainer(accelerator='cpu', max_epochs=5, logger=experiment_logger)
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)
