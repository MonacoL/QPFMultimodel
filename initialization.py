import torch.nn as nn
import torch.nn.init as init

def init_weights(m, a=0.01):  # 'a' is for Leaky ReLU
    if isinstance(m, nn.Conv2d):
        nonlinearity = 'relu'

        # Detect activation type if assigned
        if hasattr(m, 'activation'):
            if isinstance(m.activation, nn.LeakyReLU):
                nonlinearity = 'leaky_relu'
            elif isinstance(m.activation, nn.PReLU):
                nonlinearity = 'leaky_relu'  # PReLU uses same init as LeakyReLU
            elif isinstance(m.activation, nn.Tanh) or isinstance(m.activation, nn.Sigmoid):
                nonlinearity = 'sigmoid'

        init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity, a=a)
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nonlinearity = 'relu'

        if hasattr(m, 'activation'):
            if isinstance(m.activation, nn.LeakyReLU):
                nonlinearity = 'leaky_relu'
            elif isinstance(m.activation, nn.PReLU):
                nonlinearity = 'leaky_relu'  # PReLU treated like LeakyReLU
            elif isinstance(m.activation, nn.Tanh) or isinstance(m.activation, nn.Sigmoid):
                nonlinearity = 'sigmoid'

        if nonlinearity == 'leaky_relu':
            init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu', a=a)
        elif nonlinearity == 'relu':
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
        else:  # Xavier for tanh/sigmoid
            init.xavier_uniform_(m.weight)

        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

    elif isinstance(m, nn.PReLU):
        # Initialize PReLU's learnable parameter (slope) to 0.25
        init.constant_(m.weight, 0.25)