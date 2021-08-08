import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from lib.models.backbone.ResNet import ResNet_DAN_Scene_2D
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
             BasicConvLayer(512,512,(5,3),(3,1),(2,1)), #(512,3,128) ->(512,1,128)
        )
        # 2. 1 layers'  DownSample U-Net
        self.conv = nn.Sequential(
             BasicConvLayer(512,1,(1,3),(1,1),(0,1)), #(512,1,128) ->(1,1,128)
        )
        #reshape -> (1,1,128) -> (128,1,1)
        # 3. 7 layers' Upsample U-Net
        self.deconv = nn.Sequential(
             BasicDeconvLayer(128,256,[1,4],[3,2],[0,1]), # (128,1,1) -> (256,1,2)
             BasicDeconvLayer(256,256,[1,4],[3,2],[0,1]), # (256,1,2) -> (256,1,4)
             BasicDeconvLayer(256,256,[1,4],[3,2],[0,1]), # (256,1,4) -> (256,1,8)
             BasicDeconvLayer(256,256,[1,4],[3,2],[0,1]), # (256,1,8) -> (256,1,16)
             BasicDeconvLayer(256,256,[1,4],[3,2],[0,1]), # (256,1,16) -> (256,1,32)
             BasicDeconvLayer(256,256,[1,4],[3,2],[0,1]), # (256,1,32) -> (256,1,64)
             nn.Sequential(
                 nn.ConvTranspose2d(256,maxT,[1,4],[3,2],[0,1]), # (256,1,64) -> (maxT,1,128)
                 nn.BatchNorm2d(maxT),
                 nn.Sigmoid()) # To calculate attention score
        )

    def forward(self, features):
        ''' 
        features is a list of backbone features,
        shape [(32,48,512), (64,24,512), (128,12,256), (256,6,128), (512,3,128), (512,1,128)] for IAM
        '''
        # feature cascade
        x = features[0]
        for i in range(len(self.fpn)):
            x = self.fpn[i](x) + features[i+1]
        # DownSample U-Net
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # UpSample U-Net with concatnation
        for i in range(len(self.deconv)-1):
            x = self.deconv[i](x)
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
    def __init__(self, num_classes,input_size, hidden_size, dropout=0.7,max_decode_len=25):
        super(DTD, self).__init__()
        self.num_classes = num_classes
        self.rnn = nn.GRU(input_size+hidden_size,hidden_size,batch_first=True)
        self.fc = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size,num_classes)
                        )
        self.embeddings = nn.Embedding(num_classes+1, hidden_size) # +1 for <BOS>
        self.hidden_size = hidden_size
        self.max_decode_len = max_decode_len

    def forward(self, feature, attention, targets, lengths):
        '''
        feature: shape (B, C, H, W)
        attention: shape (B, maxT, H, W)
        targets: shape (B, maxT)
        lengths: shape (B)
        '''
        batch_size = feature.shape[0]
        # Firstly, normalize the attention map as the original code
        attention = attention / einops.rearrange(torch.einsum('b c h w -> b c', attention), 'b c -> b c 1 1') # b maxT h W
        # Then Compute the context matrix
        context = torch.einsum('b c h w,b t h w -> b t c',feature, attention) # -> b t c
        context = F.dropout(context, p=0.3,training=self.training)
        state = torch.zeros(1, batch_size, self.hidden_size).fill_(self.num_classes).to(device)
        outputs = []
        for i in range(max(lengths).item()):
            if i == 0:
                prev_hidden = torch.zeros((batch_size)).fill_(self.num_classes).long().to(device) # the last one is used as the <BOS>.
            else :
                prev_hidden = targets[:,i-1]

            prev_emb = self.embeddings(prev_hidden)
            y_prev = torch.cat((context[:,i,:], prev_emb), dim=1)
            output, state = self.rnn(y_prev.unsqueeze(dim=1), state)
            output = output.squeeze(1)
            outputs.append(self.fc(output))
        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs
    
    def sample(self, feature, attention):
        '''
        feature: shape (B, C, H, W)
        attention: shape (B, maxT, H, W)
        '''
        batch_size = feature.shape[0]
        # Firstly, normalize the attention map as the original code
        attention = attention / einops.rearrange(torch.einsum('b c h w -> b c', attention), 'b c -> b c 1 1') # b maxT h W
        # Then Compute the context matrix
        context = torch.einsum('b c h w,b t h w -> b t c',feature, attention) # -> b t c
        state = torch.zeros(1, batch_size, self.hidden_size).fill_(self.num_classes).to(device)
        outputs = []
        for i in range(self.max_decode_len):
            if i == 0:
                prev_hidden = torch.zeros((batch_size)).fill_(self.num_classes).long().to(device)
            else:
                prev_hidden = predicted
            prev_emb = self.embeddings(prev_hidden)
            y_prev = torch.cat((context[:,i,:], prev_emb), dim=1)
            output, state = self.rnn(y_prev.unsqueeze(dim=1), state)
            output = output.squeeze(1)
            output = self.fc(output)
            outputs.append(output)
            output = F.softmax(output, dim=1)
            _, predicted = output.max(1)
        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs

if __name__ == "__main__":
    # model = CAM_IAM(150)
    # x = torch.randn(10, 3, 192, 2048)
    # print(model(x).shape)
    attention_model = CAM_2D(25)
    backbone = ResNet_DAN_Scene_2D()
    decoder = DTD(97,backbone.out_planes, 512)

    x = torch.randn(2, 3, 32, 128)
    targets = torch.arange(50).reshape(2,25).long()
    torch.randn(2, 25).long()
    lengths = torch.IntTensor([5,6])
    
    features = backbone(x)
    attention = attention_model(features)
    res = decoder.sample(features[-1],attention)
    print(res.shape)
