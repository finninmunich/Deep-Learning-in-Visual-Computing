import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class Resnext(nn.Module):

    def __init__(self, alpha=1.0):
        super(Resnext, self).__init__()
        depths = _get_depths(alpha)
        #print(depths) # [32,16,24,40,80,96,192,320]
        resnext = torchvision.models.resnext50_32x4d(pretrained=True,progress=True)
        # for param in MNASNet.parameters():
        #     param.requires_grad = False
        self.conv1 = resnext.conv1
        self.bn1 = resnext.bn1
        self.relu = resnext.relu
        self.maxpool = resnext.maxpool

        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        depth = [256,512,1024,32,64,128]


        self.ds1 = nn.Conv2d(depth[0],depth[3],1,bias=True)
        self.ds2 = nn.Conv2d(depth[1],depth[4],1,bias=True)
        self.ds3 = nn.Conv2d(depth[2],depth[5],1,bias=True)
        self.out1 = nn.Conv2d(depth[5],depth[5],1,bias=False)
        self.out2 = nn.Conv2d(depth[5],depth[4],3,padding = 1,bias=False)
        self.out3 = nn.Conv2d(depth[5],depth[3],3,padding = 1,bias= False)
        self.inner1 = nn.Conv2d(depth[4],depth[5],1,bias=True)
        self.inner2 = nn.Conv2d(depth[3],depth[5],1,bias=True)
        # final_chs = depths[4]
        # self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True)
        # self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True)
        #
        # self.out2 = nn.Conv2d(final_chs, depths[3], 3, padding=1, bias=False)
        # self.out3 = nn.Conv2d(final_chs, depths[2], 3, padding=1, bias=False)
        # self.out_channels.append(depths[3])
        # self.out_channels.append(depths[2])
    def forward(self,x):
        #print(f"the shape of input image {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(f"the shape after conv1 {x.shape}")
        x = self.layer1(x) #(1,256,120,160)
        f1 = self.ds1(x)
        #print(f"the shape after layer1 {x.shape}")
        x = self.layer2(x)#(1,512,60,80)
        f2 = self.ds2(x)
        #print(f"the shape after layer2 {x.shape}")
        x = self.layer3(x)#(1,1024,30,40)
        f3 = self.ds3(x)
        intra_feat = f3
        outputs = []
        out = self.out1(intra_feat)
        #print(f"the shape after out1 {out.shape}")
        outputs.append(out)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(f2)  # upsample
        out = self.out2(intra_feat)
        #print(f"the shape after out2 {out.shape}")
        outputs.append(out)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(f1)
        out = self.out3(intra_feat)
        #print(f"the shape after out3 {out.shape}")
        outputs.append(out)
        #print(f"the shape after layer3 {x.shape}")


        return outputs[::-1]
    # def forward(self, x):
    #     # bn,view,c,w,h= x.shape
    #     # x = x.reshape(bn*view, c, w, h)
    #     conv0 = self.conv0(x)
    #     conv1 = self.conv1(conv0)
    #     conv2 = self.conv2(conv1)
    #
    #     intra_feat = conv2
    #     outputs = []
    #     out = self.out1(intra_feat)
    #     outputs.append(out)
    #
    #     intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
    #     out = self.out2(intra_feat)
    #     outputs.append(out)
    #
    #     intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
    #     out = self.out3(intra_feat)
    #     outputs.append(out)
    #
    #     return outputs[::-1]

if __name__ == '__main__':
    testnet = Resnext()
    x = torch.rand(1, 3, 480, 640)
    features = testnet(x)
    features = testnet(x)
    #print(features.shape)
    # features = [testnet(image) for image in imgs]
    print(features[0].shape) # 1,32,120,160 fine
    print(features[1].shape) # 1,64,60,80
    print(features[2].shape) # 1,128,30,40   coarse