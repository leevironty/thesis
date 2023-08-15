from torch_geometric.data import HeteroData
import lightning
import torch_geometric.nn as tnn
import torch


class Demo(lightning.LightningModule):
    def __init__(
        self,
        in_channels_node: int = 15,
        edge_dim: int = 15,
        out_channels: int = 15,
        num_layers: int = 10,
        num_heads: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.predictor = tnn.models.MLP(
            [out_channels * 2, out_channels * 2, 1],
        )
        self.model = tnn.models.GAT(
            v2=True,
            in_channels=in_channels_node,
            hidden_channels=out_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            heads=num_heads,
            edge_dim=edge_dim,
        )
        self.embed_event = tnn.Linear(
            in_channels=5,
            out_channels=in_channels_node,
        )
        self.embed_stop = tnn.Linear(
            in_channels=1,
            out_channels=in_channels_node,
        )
        self.embed_activity = tnn.Linear(
            in_channels=8,
            out_channels=edge_dim,
        )
        self.embed_activity_reverse = tnn.Linear(
            in_channels=8,
            out_channels=edge_dim,
        )
        self.embed_od = tnn.Linear(
            in_channels=1,
            out_channels=edge_dim,
        )
        self.embed_event_belongs_stop = tnn.Linear(
            in_channels=1,
            out_channels=edge_dim,
        )
        self.embed_stop_has_event = tnn.Linear(
            in_channels=1,
            out_channels=edge_dim,
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = torch.concat(
            [
                self.embed_event(data['event'].x),
                self.embed_stop(data['stop'].x),
            ]
        )
        device = x.device
        stop_offset = data['event'].x.shape[-2]
        adj = torch.concat(
            [
                data['demand'].edge_index + stop_offset,
                data['belongs'].edge_index
                + torch.tensor([[0], [stop_offset]]).to(device),
                data['has'].edge_index + torch.tensor([[stop_offset], [0]]).to(device),
                data['routes'].edge_index,
                data['routes_reverse'].edge_index,
            ],
            dim=-1,
        )
        edge_attr = torch.concat(
            [
                self.embed_od(data['demand'].edge_attr),
                self.embed_event_belongs_stop(data['belongs'].edge_attr),
                self.embed_stop_has_event(data['has'].edge_attr),
                self.embed_activity(data['routes'].edge_attr),
                self.embed_activity_reverse(data['routes_reverse'].edge_attr),
            ]
        )
        x = self.model(x, adj, edge_attr=edge_attr)
        x = torch.cat(
            [
                torch.index_select(x, dim=0, index=data['target'].edge_index[0, :]),
                torch.index_select(x, dim=0, index=data['target'].edge_index[1, :]),
            ],
            dim=-1,
        )

        return self.predictor(x)

    def training_step(self, batch: HeteroData, _):
        y = self(batch)
        loss = torch.nn.functional.mse_loss(y, batch['target'].weight.reshape(-1, 1))
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

    def validation_step(self, batch: HeteroData, _):
        y = self(batch)
        loss = torch.nn.functional.mse_loss(y, batch['target'].weight.reshape(-1, 1))
        self.log(
            'val_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
