import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DoubleConv(nn.Layer):
    
    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__() 
        self.double_conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Layer):
    
    def __init__(self, in_channels, out_channels):

        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2D(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Layer):

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = paddle.concat([x1, x2], axis=1)
        
        return self.conv(x)

class OutConv(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class Net(nn.Layer):
    def __init__(self,n_channels):
        super(Net, self).__init__()
        self.n_channels = n_channels   #BGR
        
        ch = [16, 32, 64, 128, 256]
        
        self.inc = DoubleConv(self.n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4]//2)

        self.up1 = Up(ch[4], ch[3]//2)
        self.up2 = Up(ch[3], ch[2]//2)
        self.up3 = Up(ch[2], ch[1]//2)
        self.up4 = Up(ch[1], ch[0])

        self.outc = OutConv(ch[0], 3)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

if __name__ == '__main__':

    net = Net(3).eval()
    img = paddle.zeros([1,3,320,320])
    out = Net(img)

