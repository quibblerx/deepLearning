import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, global_add_pool, GATConv

class GCN(torch.nn.Module):
    # I added default values here so your old code calls won't crash
    def __init__(self, num_node_features, hidden_channels=256, num_layers=5, use_batchnorm=True, dropout=0.3):
        super(GCN, self).__init__()
        
        
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        
        # A list to hold however many layers you want
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Layer 1 (Input)
        self.convs.append(SAGEConv(num_node_features, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))

        # Layers 2 to N (Hidden) - This loop is what lets you change depth!
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        # Output Layer
        self.linear = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Loop through the layers defined above
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # Apply Batch Norm if enabled
            if self.use_batchnorm:
                x = self.bns[i](x)
            
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout
        x = global_mean_pool(x, batch)
        

        
        # Final prediction
        x = self.linear(x)

        return x.squeeze(-1)
