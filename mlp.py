import torch
from torch import nn
import initialization as intz #initialization custom class

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features:int, out_features:int, n_layers: int, n_neurons: int, batch_norm:bool, p_dropout: float, kernel_size:int, stride_pool: int, padding_pool:int, mask: torch.Tensor, device: str):
        super().__init__()
        self.mask=mask #mask for the points in the Area of Interest
        self.device=device #cpu or gpu
        self.n_neurons=n_neurons # number of neurons of hidden layers
        self.avg_pool=nn.AvgPool2d(kernel_size=kernel_size, stride=stride_pool, padding=padding_pool) #average pool layer
        self.n_layers=n_layers #number of total layers. the number of hidden layers then is n_layers-2
        self.batch_norm=batch_norm #boolean for batch normalization request
        self.p_dropout=p_dropout #boolean for dropout request

        self.layers = nn.ModuleList()   
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.prelus= nn.ModuleList()

        for i in range(self.n_layers):
            in_neurons = in_features if i == 0 else n_neurons #check for first layer and define appropriate number of neurons
            out_neurons = out_features if i == n_layers-1 else n_neurons #check for last layer and define appropriate number of neurons
            self.layers.append(nn.Linear(in_neurons, out_neurons))
            self.dropouts.append(nn.Dropout(p=self.p_dropout)) #provide dropout in between layers if necessary
            self.batch_norms.append(nn.BatchNorm1d(out_features if i == n_layers-1 else n_neurons)) #provide batch normalization in between layers if necessary
            self.prelus.append(nn.PReLU(num_parameters=1)) #PReLU allows for better bias correction between layers

        self.apply(lambda m: intz.init_weights(m, a=0.01)) #weights initialization based of layer type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (B,in_features,H*W)
        # where B is the batch size, in_features is the number of models to be multimodelled, H and W are shapes of the data grids
        # H*W means the grids have been flatted before been fed to the network

        for i,(layer, dropout, batchnorm, prelu) in enumerate(zip(self.layers, self.dropouts, self.batch_norms, self.prelus)):
            x = layer(x)
            if i==self.n_layers-1:
                if self.batch_norm: #check for batch normalization request
                    x=batchnorm(x)
            x = prelu(x)
            if i==self.n_layers-1:
                if self.p_dropout>0: #check for dropout request
                    x=dropout(x)

        # x -> (B,H*W) the output is still flattened
        
        self.output=torch.zeros(x.size(0),self.mask.size(0),self.mask.size(1), dtype=torch.float).to(self.device) # -> (B,H,W)
        self.output[:,self.mask]=torch.relu(x) #put the flattened output in an output grid using the Area of Interest mask
        #relu ensures no negative precipitation is permitted

        return self.avg_pool(self.output) #apply average pool to the output,to avoid salt and pepper noise
            