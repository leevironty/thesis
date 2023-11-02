from turtle import forward
from typing import Any, Literal
import torch_geometric
from torch_geometric.data import HeteroData, Batch
from torch_geometric.typing import NodeType
import lightning

# import torch_geometric.nn as tnn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, HeteroLayerNorm, HeteroBatchNorm, BatchNorm, LayerNorm, DenseGATConv, MLP
from torch_geometric.transforms import AddRandomWalkPE, AddLaplacianEigenvectorPE, AddMetaPaths
from torch_geometric.nn.module_dict import ModuleDict
import torch
from torch_geometric.typing import Metadata
from torch_geometric.utils import segment
from torch_geometric.nn.dense import Linear



class HeteroGraphEmbedding(torch.nn.Module):
    def __init__(self, out_channels: int, meta: Metadata):
        super().__init__()
        self.maps = ModuleDict()
        for key in meta[0]:
            self.maps[key] = Linear(-1, out_channels)
    
    def forward(self, x_dict: dict[str, torch.Tensor], ptr_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        def convert(key: str) -> torch.Tensor:
            x = segment(x_dict[key], ptr_dict[key], reduce='max')
            return self.maps[key](x)
        
        return torch.concat([convert(key) for key in x_dict.keys()]).max(dim=0).values.relu()




class Norm(torch.nn.Module):
    def __init__(self, in_channels: int, meta: Metadata):
        super().__init__()
        self.norms = ModuleDict()
        for node_type in meta[0]:
            self.norms[node_type] = BatchNorm(in_channels)
            # self.norms[node_type] = LayerNorm(in_channels, mode='node')
    
    def forward(self, x_dict: dict[NodeType, torch.Tensor]) -> dict[NodeType, torch.Tensor]:
        return {
            node_type: self.norms[node_type](x)
            for node_type, x in x_dict.items()
        }


# def add_full_connections(data: HeteroData) -> HeteroData:


class Predictor(lightning.LightningModule):
    def __init__(
        self,
        lr: float,
        hidden_channels: int,
        num_heads: int,
        num_layers: int,
        data: HeteroData,
        # norm: Literal['batch'] | Literal['layer']
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # self.add_metapaths = AddMetaPaths(metapaths)

        # data_aug = self.add_metapaths(data)
        # meta = data_aug.metadata()
        meta = data.metadata()

        self.lr = lr
        # self.pe = AddRandomWalkPE(walk_length=10)
        self.pe = AddLaplacianEigenvectorPE(k=4)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        # self.aggregator = HeteroGraphEmbedding(out_channels=hidden_channels, meta=meta)
        # self.agg_map = Linear(hidden_channels, hidden_channels)
        self.embedding_mapper = MLP([hidden_channels, hidden_channels, 1])
        # self.fully_connected = DenseGATConv(
        #     in_channels=hidden_channels,
        #     out_channels=hidden_channels,
        #     heads=4,
        # )
        self.node_embed_mapping = torch.nn.Linear(hidden_channels, hidden_channels)
        for _ in range(num_layers):
            layer = HGTConv(
                in_channels=-1,
                out_channels=hidden_channels,
                metadata=meta,
                heads=num_heads,
                group='sum',
            )
            self.convs.append(layer)
        
        for _ in range(num_layers - 1):
            layer = Norm(
                in_channels=hidden_channels,
                meta=meta,
            )
            self.norms.append(layer)


    def forward(self, data: HeteroData) -> torch.Tensor:
        device = data['route_features'].x.device
        # data = self.add_metapaths(data)
        homogenous = data.to_homogeneous(node_attrs=[])
        encodings = self.pe(homogenous)
        for key in encodings.keys:
            if key != 'metapath_dict':
                encodings[key] = encodings[key].to(device)
        hetero_encodings = encodings.to_heterogeneous()
        x = {
            node_type: torch.cat(
                [data[node_type].x, hetero_encodings[node_type].laplacian_eigenvector_pe],
                dim=-1
            )
            for node_type in data.node_types
        }
        # x = data.x_dict
        # for node_type in data.node_types:
        #     data[node_type].x = torch.cat(
        #         [data[node_type].x, hetero_encodings[node_type].laplacian_eigenvector_pe],
        #         dim=-1
        #     )

        # x = data.x_dict
        edges = data.edge_index_dict
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edges)
            x = norm(x)
        
        # one less norm that conv -> last conv needs to be applied separately
        x = self.convs[-1](x, edges)
        # x = x['route_features']
        # #
        # x = self.fully_connected(x, adj=torch.ones())
        # graph_embedding = self.aggregator(x, data.ptr_dict).reshape(1, -1)
        # adt = self.agg_map(graph_embedding)
        out = self.node_embed_mapping(x['route_features'])  # + adt
        out = self.embedding_mapper(out)
        return out * data['route_features'].mask
        # return self.last(x['route_features']) * data['route_features'].mask

    def predict(self, data: HeteroData) -> torch.Tensor:
        return self(data).relu()

    def training_step(self, data: Batch, _) -> torch.Tensor:
        # scheduler = self.sch
        y = data['route_features'].target
        # zero_mask = y == 0
        # y = y - 1 * zero_mask
        y_hat = self(data)
        loss = F.mse_loss(y_hat, y)
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=data.num_graphs,
        )
        return loss

    def validation_step(self, data: Batch, _) -> torch.Tensor:
        y = data['route_features'].target
        # zero_mask = y == 0
        # y = y - 1 * zero_mask
        y_hat = self(data)
        loss = F.mse_loss(y_hat, y)
        self.log(
            'val_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=data.num_graphs,
        )
        return loss

    def predict_step(self, data: HeteroData, _):
        return self.predict(data)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer=optimizer,
    #         T_max=6000,
    #     )
    #     return [optimizer], [scheduler]
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,
            threshold=0.2,
            threshold_mode='rel',
            patience=1000,
            cooldown=500,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_step",
                "frequency": 1,
                "interval": "step",
                "name": 'plateau-scheduler',
            },
        }
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.3,
        #     threshold=0.2,
        #     threshold_mode='rel',
        #     patience=50,
        #     cooldown=150,
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "train_loss_epoch",
        #         "frequency": 1,
        #         "interval": "epoch",
        #         "name": 'plateau-scheduler',
        #     },
        # }
