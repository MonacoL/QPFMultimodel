import torch
import torch.nn as nn
import initialization as intz

def conv_block(in_channels, out_channels, use_batchnorm=False):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"))
    if use_batchnorm: #check for batch normalization request
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.PReLU(num_parameters=1)) #PReLU allows for better bias correction between layers
    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_batchnorm: bool = False):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        # Encoder Blocks
        self.enc1 = nn.Sequential(
            conv_block(in_features, 64, use_batchnorm),
            conv_block(64, 64, use_batchnorm)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            conv_block(64, 128, use_batchnorm),
            conv_block(128, 128, use_batchnorm)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(
            conv_block(128, 256, use_batchnorm),
            conv_block(256, 256, use_batchnorm)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = nn.Sequential(
            conv_block(256, 512, use_batchnorm),
            conv_block(512, 512, use_batchnorm)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = nn.Sequential(
            conv_block(512, 1024, use_batchnorm),
            conv_block(1024, 1024, use_batchnorm)
        )

        # Decoder Blocks
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            conv_block(1024, 512, use_batchnorm),
            conv_block(512, 512, use_batchnorm)
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            conv_block(512, 256, use_batchnorm),
            conv_block(256, 256, use_batchnorm)
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            conv_block(256, 128, use_batchnorm),
            conv_block(128, 128, use_batchnorm)
        )

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            conv_block(128, 64, use_batchnorm),
            conv_block(64, 64, use_batchnorm)
        )

        # Output layer
        self.outconv = nn.Conv2d(64, out_features, kernel_size=1)
        
        # Custom initialization if necessary
        self.apply(lambda m: intz.init_weights(m, a=0.01))

    def forward(self, x):
        # input (B,in_features,H,W)
        # where B is the batch size, in_features is the number of models to be multimodelled, H and W are shapes of the data grids
        # H and W must multiple of 16, use zero padding in case

        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        x4 = self.enc4(p3)
        p4 = self.pool4(x4)

        center = self.center(p4)

        # Decoder with skip connections
        up1 = self.upconv1(center)
        cat1 = torch.cat([up1, x4], dim=1)
        d1 = self.dec1(cat1)

        up2 = self.upconv2(d1)
        cat2 = torch.cat([up2, x3], dim=1)
        d2 = self.dec2(cat2)

        up3 = self.upconv3(d2)
        cat3 = torch.cat([up3, x2], dim=1)
        d3 = self.dec3(cat3)

        up4 = self.upconv4(d3)
        cat4 = torch.cat([up4, x1], dim=1)
        d4 = self.dec4(cat4)

        out = torch.relu(self.outconv(d4)) # -> (B,1,H,W)
        return out.squeeze(1) # -> (B,H,W)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # If input and output channels differ, adjust the residual.
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv_block(x)
        residual = self.shortcut(x)
        out += residual
        return self.prelu(out)

class ResUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_filters=32):
        super(ResUNet, self).__init__()
        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base_filters*2, base_filters*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(base_filters*4, base_filters*8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(base_filters*8, base_filters*16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters*16, base_filters*8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_filters*16, base_filters*8)  # Concatenated with enc4

        self.up3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_filters*8, base_filters*4)   # Concatenated with enc3

        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_filters*4, base_filters*2)   # Concatenated with enc2

        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_filters*2, base_filters)     # Concatenated with enc1

        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.out_activation = nn.ReLU()  # Enforce non-negative outputs

        self.apply(lambda m: intz.init_weights(m, a=0.01))

    def forward(self, x):
        # input (B,in_features,H,W)
        # where B is the batch size, in_features is the number of models to be multimodelled, H and W are shapes of the data grids
        # H and W must multiple of 16, use zero padding in case

        # Encoder
        enc1 = self.enc1(x)                          
        enc2 = self.enc2(self.pool1(enc1))             
        enc3 = self.enc3(self.pool2(enc2))             
        enc4 = self.enc4(self.pool3(enc3))            
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with concatenation.
        dec4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([enc4, dec4], dim=1))

        dec3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([enc3, dec3], dim=1))

        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([enc2, dec2], dim=1))

        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([enc1, dec1], dim=1))
  
        out = self.final_conv(dec1)
        out = self.out_activation(out) # -> (B,1,H,W)

        return out.squeeze(1)  # -> (B,H,W)