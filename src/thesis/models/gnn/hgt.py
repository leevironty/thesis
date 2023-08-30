from torch_geometric.data import HeteroData
import lightning

# import torch_geometric.nn as tnn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
import torch
from torch_geometric.typing import Metadata


class Predictor(lightning.LightningModule):
    def __init__(
        self,
        lr: float,
        hidden_channels: int,
        num_heads: int,
        num_layers: int,
        meta: Metadata,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.convs = torch.nn.ModuleList()
        self.last = torch.nn.Linear(hidden_channels, 1)
        for _ in range(num_layers):
            layer = HGTConv(
                in_channels=-1,
                out_channels=hidden_channels,
                metadata=meta,
                heads=num_heads,
                group='sum',
            )
            self.convs.append(layer)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = data.x_dict
        edges = data.edge_index_dict
        for layer in self.convs:
            x = layer(x, edges)
        return self.last(x['route_features']) * data['route_features'].mask

    def predict(self, data: HeteroData) -> torch.Tensor:
        return self(data).relu()

    def training_step(self, data: HeteroData, _) -> torch.Tensor:
        y = data['route_features'].target
        y_hat = self(data)
        loss = F.mse_loss(y_hat, y)
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )
        return loss

    def validation_step(self, data: HeteroData, _) -> torch.Tensor:
        y = data['route_features'].target
        y_hat = self(data)
        loss = F.mse_loss(y_hat, y)
        self.log(
            'val_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )
        return loss

    def predict_step(self, data: HeteroData, _):
        return self.predict(data)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
