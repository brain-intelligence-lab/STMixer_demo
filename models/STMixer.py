import torch
import torch.nn as nn
from models.spiking_layer import LIFSpike, ExpandTime
from models.surrogate_module import SurrogateModule

class SPSV2(nn.Module):
    def __init__(self, img_size=128, downsample_times=4, in_channels=3, embd_dims=256, T=1):
        super(SPSV2, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.downsample_times = downsample_times
        self.T = T
        self.main_embd_dims = (embd_dims // 8) * 7
        self.short_embd_dims = (embd_dims // 8) * 1

        if downsample_times == 2:
            self.proj_conv = nn.Sequential(
                nn.Conv2d(in_channels, embd_dims // 8, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embd_dims // 8),
                LIFSpike(T=T),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(embd_dims // 8, embd_dims // 4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embd_dims // 4),
                LIFSpike(T=T),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(embd_dims // 4, embd_dims // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embd_dims // 2),
                LIFSpike(T=T),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(embd_dims // 2, embd_dims, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embd_dims),
                LIFSpike(T=T),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(embd_dims, self.main_embd_dims, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.main_embd_dims),
            )
        elif downsample_times == 4:
            self.proj_conv = nn.Sequential(
                nn.Conv2d(in_channels, embd_dims // 8, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embd_dims // 8),
                LIFSpike(T=T),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(embd_dims // 8, embd_dims // 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embd_dims // 4),
                LIFSpike(T=T),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(embd_dims // 4, embd_dims // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embd_dims // 2),
                LIFSpike(T=T),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(embd_dims // 2, embd_dims, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embd_dims),
                LIFSpike(T=T),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(embd_dims, self.main_embd_dims, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.main_embd_dims),
            )
        else:
            raise NotImplementedError

        short_stride = 2 ** downsample_times
        self.short_proj1 = nn.Sequential(
            nn.Conv2d(in_channels, self.short_embd_dims, kernel_size=short_stride,
                      stride=short_stride, padding=0, bias=False),
            nn.BatchNorm2d(self.short_embd_dims),
        )

    def forward(self, x):
        short_x1 = self.short_proj1(x)
        x = self.proj_conv(x)
        x = torch.cat([short_x1, x], dim=1)
        x = x.flatten(-2)
        return x

class STM(nn.Module):
    def __init__(self, feature_dim, num_dim, num_head, T):
        super().__init__()
        self.num_head = num_head
        self.num_dim = num_dim
        self.feature_dim = feature_dim
        self.num_group = feature_dim // num_head
        self.linear_mix = nn.ModuleList([nn.Linear(num_dim, num_dim, bias=False) for _ in range(num_head)])
        # init the weights
        for m in self.linear_mix:
            if isinstance(m, nn.Linear):
                # kaiming init
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = x.reshape(x.size(0), self.num_head, self.num_group, -1)
        y = torch.zeros_like(x)
        for i in range(self.num_head):
            y[:, i, :, :] = self.linear_mix[i](x[:, i, :, :])
        x = y.reshape(x.size(0), self.feature_dim, -1)
        return x

class SSA(nn.Module):
    def __init__(self, feature_dim, num_dim, num_head, T):
        super(SSA, self).__init__()
        self.num_head = num_head
        self.num_dim = num_dim
        self.feature_dim = feature_dim
        self.T = T
        self.group_dim = feature_dim // num_head
        self.Wq = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feature_dim),
            LIFSpike(T=T),
        )
        self.Wk = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feature_dim),
            LIFSpike(T=T),
        )
        self.Wv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feature_dim),
            LIFSpike(T=T),
        )
        self.proj = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feature_dim),
        )
        self.act = LIFSpike(T=T)
        self.scale = 0.125

    def forward(self, x):
        B, C, N = x.size()
        q = self.Wq(x) # [T*B, C, N]
        k = self.Wk(x) # [T*B, C, N]
        v = self.Wv(x) # [T*B, C, N]
        q = q.reshape(B, self.num_head, self.group_dim, -1)
        k = k.reshape(B, self.num_head, self.group_dim, -1)
        v = v.reshape(B, self.num_head, self.group_dim, -1)
        q = q.permute(0, 1, 3, 2) # [B, num_head, N, group_dim]
        v = v.permute(0, 1, 3, 2) # [B, num_head, N, group_dim]
        attn = torch.matmul(q, k) * self.scale # [B, num_head, N, N]
        x = torch.matmul(attn, v) # [B, num_head, N, group_dim]
        x = x.permute(0, 1, 3, 2).reshape(B, C, N)
        x = self.act(x)
        x = self.proj(x)
        return x

class TokenMixer(nn.Module):
    def __init__(self, feature_dim, num_patches, num_head,  T):
        super(TokenMixer, self).__init__()
        self.T = T
        ratio = 1
        self.mid_dim1 = int(feature_dim * ratio)
        self.fc1 = nn.Sequential(
            LIFSpike(T=T),
            nn.Conv1d(feature_dim, self.mid_dim1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.mid_dim1),
            LIFSpike(T=T),
        )
        self.attn = STM(self.mid_dim1, num_patches, num_head, T) # 
        # self.attn = SSA(self.mid_dim1, num_patches, num_head, T) # change to standard SSA module
        self.fc2 = nn.Sequential(
            LIFSpike(T=T),
            nn.Conv1d(self.mid_dim1, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.attn(x)
        x = self.fc2(x)

        return x

class FFN(nn.Module):
    def __init__(self, feature_dim, ratio, T):
        super().__init__()
        self.feature_dim = feature_dim
        self.ratio = ratio
        self.T = T
        self.mid_dim = int(feature_dim * ratio)
        self.fc1 = nn.Sequential(
            LIFSpike(T=T),
            nn.Conv1d(feature_dim, self.mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.mid_dim),
        )
        self.fc2 = nn.Sequential(
            LIFSpike(T=T),
            nn.Conv1d(self.mid_dim, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, feature_dim, num_pathes, ratio, num_head, T):
        super().__init__()
        self.token_mix = TokenMixer(feature_dim, num_pathes, num_head, T)
        self.channel_mix = FFN(feature_dim, ratio, T)

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

class STMixerV3(nn.Module):
    def __init__(self, img_size=128, downsample_times=4, in_channels=3, embd_dims=256,
                 T=1, mlp_ratio=2, depths=6, num_head=8, num_classes=100, sml=False):
        super(STMixerV3, self).__init__()
        self.img_size = img_size
        self.T = T
        self.img_size = img_size
        self.depths = depths
        self.sml = sml
        self.HW = img_size // (2 ** downsample_times)
        self.num_patches = self.HW ** 2
        self.in_channels = in_channels
        if self.in_channels == 3:
            self.expand = ExpandTime(T=T)
        self.patch_embd = SPSV2(img_size=img_size, downsample_times=downsample_times,
                              in_channels=in_channels, embd_dims=embd_dims, T=T)

        self.token_dim = int(self.num_patches * 1.0)
        self.block = nn.ModuleList(
            [Encoder(embd_dims, self.token_dim, mlp_ratio, num_head, T) for _ in range(depths)]
        )

        self.head = nn.Linear(embd_dims, num_classes)


        # init the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # if self.sml and depths == 2:
        #     self.sml1 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=1, T=T)
        # elif self.sml and depths == 4:
        #     self.sml1 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=2, T=T)
        #     self.sml2 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=1, T=T)
        #
        # elif self.sml and depths == 6:
        #     self.sml1 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=2, T=T)
        #     self.sml2 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=1, T=T)
        #     self.sml3 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=1, T=T)
        # elif self.sml and depths == 8:
        #     self.sml1 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=3, T=T)
        #     self.sml2 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=2, T=T)
        #     self.sml3 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=1, T=T)
        #     self.sml4 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=1, T=T)
        # elif self.sml and depths >= 10:
        #     self.sml1 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=3, T=T)
        #     self.sml2 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=2, T=T)
        #     self.sml3 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=1, T=T)
        #     self.sml4 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=1, T=T)
        #     self.sml5 = SurrogateModule(embd_dims, embd_dims, self.num_patches, num_classes, num_head, num_layer=1, T=T)

        max_sml_layer_dict = {2: 1, 4: 2, 6: 3, 8: 3, 10: 3}

        if self.sml and depths >= 2:
            sml_num = depths // 2
            if depths % 2 != 0:
                depths -= 1
            max_sml_layer = max_sml_layer_dict[depths]
            for i in range(sml_num):
                num_layer = max(max_sml_layer - i, 1)
                setattr(self, f'sml{i+1}', SurrogateModule(embd_dims, embd_dims, self.num_patches,
                                                           num_classes, num_head, num_layer=num_layer, T=T))

    def forward_sdt(self, x):
        x = self.patch_embd(x)
        for blk in self.block:
            x = blk(x)
        x = x.mean(dim=-1)
        x = x.reshape(self.T, -1, x.shape[-1])
        x = self.head(x)
        x = x.mean(dim=0)
        return x

    def forward_sp(self, x):
        outs = []
        # x = self.expand(x)
        x = self.patch_embd(x)
        outs.append(self.sml1(x))
        for bi, blk in enumerate(self.block):
            x = blk(x)
            if bi == 1 and self.depths >= 4:
                outs.append(self.sml2(x))
            if bi == 3 and self.depths >= 6:
                outs.append(self.sml3(x))
            if bi == 5 and self.depths >= 8:
                outs.append(self.sml4(x))
            if bi == 7 and self.depths >= 10:
                outs.append(self.sml5(x))
        x = x.mean(dim=-1)
        x = x.reshape(self.T, -1, x.shape[-1])
        x = self.head(x)
        x = x.mean(dim=0)
        outs.insert(0, x)
        return outs

    def forward(self, x):
        if self.in_channels == 3:
            x = self.expand(x)
        elif self.in_channels == 2:
            x = x.permute(1, 0, 2, 3, 4)
            x = x.reshape(-1, 2, self.img_size, self.img_size)
        if self.sml:
            return self.forward_sp(x)
        else:
            return self.forward_sdt(x)


if __name__ == '__main__':
    x = torch.randn(4,3,32,32)
    model = STMixerV3(img_size=32, downsample_times=2, in_channels=3,
                      embd_dims=256, T=2, mlp_ratio=4, depths=4,
                      num_head=16, num_classes=100, sml=True)
    with torch.no_grad():
        y = model(x)
        print(len(y))
        print(y[0].shape)