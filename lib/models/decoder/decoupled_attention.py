import torch
import torch.nn as nn

"""To better understanding CAM, I Have to fix all the model parameters"""
class BasicConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class BasicDeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicDeconvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

"""To better understanding CAM, I Have to fix all the model parameters"""
class CAM_IAM(nn.Module):
    def __init__(self, maxT):
        super(CAM_IAM, self).__init__()
        # 1. cascade multiscale features
        self.fpn = nn.Sequential(
             BasicConvLayer(32,64,(3,3),(2,1),(1,1)), #(32,48,512) ->(64,24,512)
             BasicConvLayer(64,128,(3,3),(2,2),(1,1)), #(64,24,512) ->(128,12,256)
             BasicConvLayer(128,256,(3,3),(2,2),(1,1)), #(128,12,256) ->(256,6,128)
             BasicConvLayer(256,512,(3,3),(2,1),(1,1)), #(256,6,128) ->(512,3,128)
             BasicConvLayer(512,256,(5,3),(3,1),(2,1)), #(512,3,128) ->(256,1,128)
        )
        # 2. 7 layers'  DownSample U-Net
        self.conv = nn.Sequential(
             BasicConvLayer(256,128,(3,3),(1,2),(1,1)), #(256,1,128) ->(128,1,64)
             BasicConvLayer(128,128,(3,3),(1,2),(1,1)), #(128,1,64) -> (128,1,32)
             BasicConvLayer(128,128,(3,3),(1,2),(1,1)), #(128,1,32) -> (128,1,16)
             BasicConvLayer(128,128,(3,3),(1,2),(1,1)), #(128,1,16) -> (128,1,8)
             BasicConvLayer(128,128,(3,3),(1,2),(1,1)), #(128,1,8) -> (128,1,4)
             BasicConvLayer(128,128,(3,3),(1,2),(1,1)), #(128,1,4) -> (128,1,2)
             BasicConvLayer(128,128,(3,3),(1,2),(1,1)), #(128,1,2) -> (128,1,1)
        )

        # 3. 7 layers' Upsample U-Net
        self.deconv = nn.Sequential(
             BasicDeconvLayer(128,128,[1,4],[1,2],[0,1]), # (128,1,1) -> (128,1,2)
             BasicDeconvLayer(128,128,[1,4],[1,2],[0,1]), # (128,1,2) -> (128,1,4)
             BasicDeconvLayer(128,128,[1,4],[1,2],[0,1]), # (128,1,4) -> (128,1,8)
             BasicDeconvLayer(128,128,[1,4],[1,2],[0,1]), # (128,1,8) -> (128,1,16)
             BasicDeconvLayer(128,128,[1,4],[1,2],[0,1]), # (128,1,16) -> (128,1,32)
             BasicDeconvLayer(128,128,[1,4],[1,2],[0,1]), # (128,1,32) -> (128,1,64)
             nn.Sequential(
                 nn.ConvTranspose2d(128,maxT,[1,4],[1,2],[0,1]), # (128,1,64) -> (maxT,1,128)
                 nn.BatchNorm2d(maxT),
                 nn.Sigmoid()) # To calculate attention score
        )

    def forward(self, features):
        ''' 
        features is a list of backbone features,
        shape [(32,48,512), (64,24,512), (128,12,256), (256,6,128), (512,3,128), (256,1,128)] for IAM
        '''
        # feature cascade
        x = features[0]
        for i in range(len(self.fpn)):
            x = self.fpn[0](x) + features[i+1]
        # DownSample U-Net
        DownSample_features = []
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            DownSample_features.append(x)
        # UpSample U-Net with concatnation
        for i in range(len(self.deconv)-1):
            x = self.deconv[i](x)
            x = x + DownSample_features[-(i+2)]
        x = self.deconv[-1](x)
        return x

class CAM_1D(nn.Module):
    def __init__(self, maxT):
        super(CAM_1D, self).__init__()
        # 1. cascade multiscale features
        self.fpn = nn.Sequential(
             BasicConvLayer(32,64,(3,3),(2,2),(1,1)), #(32,16,64) ->(64,8,32)
             BasicConvLayer(64,128,(3,3),(2,1),(1,1)), #(64,8,32) ->(128,4,32)
             BasicConvLayer(128,256,(3,3),(2,1),(1,1)), #(128,4,32) ->(256,2,32)
             BasicConvLayer(256,512,(3,3),(2,1),(1,1)), #(256,2,32) ->(512,1,32)
        )
        # 2. 4 layers'  DownSample U-Net
        self.conv = nn.Sequential(
             BasicConvLayer(512,64,(3,3),(1,2),(1,1)), #(512,1,32) ->(64,1,16)
             BasicConvLayer(64,64,(3,3),(2,2),(1,1)), #(64,1,16) -> (64,1,8)
             BasicConvLayer(64,64,(3,3),(2,2),(1,1)), #(64,1,8) -> (64,1,4)
             BasicConvLayer(64,64,(3,3),(2,2),(1,1)), #(64,1,4) -> (64,1,2)
        )

        # 3. 4 layers' Upsample U-Net
        self.deconv = nn.Sequential(
             BasicDeconvLayer(64,64,[1,4],[1,2],[0,1]), # (64,1,2) -> (64,1,4)
             BasicDeconvLayer(64,64,[1,4],[1,2],[0,1]), # (64,1,4) -> (64,1,8)
             BasicDeconvLayer(64,64,[1,4],[1,2],[0,1]), # (64,1,8) -> (64,1,16)
             nn.Sequential(
                 nn.ConvTranspose2d(64,maxT,[1,4],[1,2],[0,1]), # (64,1,16) -> (maxT,1,32)
                 nn.BatchNorm2d(maxT),
                 nn.Sigmoid()) # To calculate attention score
        )

    def forward(self, features):
        ''' 
        features is a list of backbone features,
        shape [(16, 64), (8, 32), (4, 32), (2, 32), (1, 32)] for 1D
        '''
        # feature cascade
        x = features[0]
        for i in range(len(self.fpn)):
            x = self.fpn[i](x) + features[i+1]
        # DownSample U-Net
        DownSample_features = []
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            DownSample_features.append(x)
        # UpSample U-Net with concatnation
        for i in range(len(self.deconv)-1):
            x = self.deconv[i](x)
            x = x + DownSample_features[-(i+2)]
        x = self.deconv[-1](x)
        return x

class CAM_2D(nn.Module):
    def __init__(self, maxT):
        super(CAM_2D, self).__init__()
        # 1. cascade multiscale features
        self.fpn = nn.Sequential(
             BasicConvLayer(32,128,(3,3),(2,2),(1,1)), #(32,16,64) ->(128,8,32)
             BasicConvLayer(128,512,(3,3),(1,1),(1,1)), #(128,8,32) ->(512,8,32)
        )
        # 2. 4 layers'  DownSample U-Net
        self.conv = nn.Sequential(
             BasicConvLayer(512,64,(3,3),(1,2),(1,1)), #(512,8,32) ->(64,8,16)
             BasicConvLayer(64,64,(3,3),(2,2),(1,1)), #(64,8,16) -> (64,4,8)
             BasicConvLayer(64,64,(3,3),(2,2),(1,1)), #(64,4,8) -> (64,2,4)
             BasicConvLayer(64,64,(3,3),(2,2),(1,1)), #(64,2,4) -> (64,1,2)
        )

        # 3. 7 layers' Upsample U-Net
        self.deconv = nn.Sequential(
             BasicDeconvLayer(64,64,[4,4],[2,2],[1,1]), # (128,1,2) -> (128,2,4)
             BasicDeconvLayer(64,64,[4,4],[2,2],[1,1]), # (128,2,4) -> (128,4,8)
             BasicDeconvLayer(64,64,[4,4],[2,2],[1,1]), # (128,4,8) -> (128,8,16)
             nn.Sequential(
                 nn.ConvTranspose2d(64,maxT,[1,4],[1,2],[0,1]), # (128,8,16) -> (maxT,8,32)
                 nn.BatchNorm2d(maxT),
                 nn.Sigmoid()) # To calculate attention score
        )

    def forward(self, features):
        ''' 
        features is a list of backbone features,
        shape [(16, 64), (8, 32), (8, 32)] for 2D
        '''
        # feature cascade
        x = features[0]
        for i in range(len(self.fpn)):
            x = self.fpn[i](x) + features[i+1]
        # DownSample U-Net
        DownSample_features = []
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            DownSample_features.append(x)
        # UpSample U-Net with concatnation
        for i in range(len(self.deconv)-1):
            x = self.deconv[i](x)
            x = x + DownSample_features[-(i+2)]
        x = self.deconv[-1](x)
        return x

""" Decoupled Text Decoder """
class DTD(nn.Module):
    pass
if __name__ == "__main__":
    # model = CAM_IAM(150)
    # x = torch.randn(10, 3, 192, 2048)
    # print(model(x).shape)
    model = CAM_2D(25)
    x = torch.randn(10, 3, 32, 128)
    print(model(x).shape)