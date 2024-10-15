import torch
import torch.nn as nn

class STM(nn.Module):
    def __init__(self, feature_dim, num_dim, num_head):
        super().__init__()
        self.num_head = num_head
        self.num_dim = num_dim
        self.feature_dim = feature_dim
        self.num_group = feature_dim // num_head
        self.linear_mix = nn.ModuleList([nn.Linear(num_dim, num_dim, bias=False) for _ in range(num_head)])
        # init the weights
        for m in self.linear_mix:
            if isinstance(m, nn.Linear):
                # m.weight.data.fill_(1.0 / num_dim)
                # kaeming init
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = x.reshape(x.size(0), self.num_head, self.num_group, -1)
        y = torch.zeros_like(x)
        for i in range(self.num_head):
            y[:, i, :, :] = self.linear_mix[i](x[:, i, :, :])
        x = y.reshape(x.size(0), self.feature_dim, -1)
        return x

class TokenMixer(nn.Module):
    def __init__(self, feature_dim, num_patches, num_head):
        super(TokenMixer, self).__init__()
        self.mid_dim1 = int(feature_dim)
        self.fc1 = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(feature_dim, self.mid_dim1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.mid_dim1),
            nn.GELU(),
        )
        self.attn = STM(self.mid_dim1, num_patches, num_head)
        # self.attn2 = LinearMix(self.mid_dim1, num_patches, num_head, T)
        self.fc2 = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(self.mid_dim1, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feature_dim),
        )


    def forward(self, x):
        x = self.fc1(x)
        x = self.attn(x)
        x = self.fc2(x)
        return x

class FFN(nn.Module):
    def __init__(self, feature_dim, expansion_factor=4):
        super(FFN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim * expansion_factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feature_dim * expansion_factor),
            nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(feature_dim * expansion_factor, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class SurrogateEncoder(nn.Module):
    def __init__(self, dims, num_patches, num_head, expansion_factor):
        super().__init__()
        self.spatial_mix = TokenMixer(dims, num_patches, num_head)
        self.channel_mix = FFN(dims, expansion_factor)

    def forward(self, x):
        x = x + self.spatial_mix(x)
        x = x + self.channel_mix(x)
        return x

class SurrogateModule(nn.Module):
    def __init__(self, in_dims, dims, num_patches, num_classes, num_head, num_layer, T):
        super().__init__()
        self.T = T
        self.proj_conv = nn.Sequential(
            nn.Conv1d(in_dims, dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(dims),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(*[SurrogateEncoder(dims, num_patches, num_head, 4)
                                       for _ in range(num_layer)])
        self.head = nn.Linear(dims, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = x.view(-1, self.T, *x.shape[-2:])
        # B, T, C, N = x.shape
        # x = x.reshape(B, T * C, N)
        x = x.mean(1)
        x = self.proj_conv(x)
        x = self.encoder(x)
        x = x.mean(-1)
        x = x.flatten(1)
        x = self.head(x)
        return x

class SDSurrogateModule(nn.Module):
    def __init__(self, in_dims, dims, num_patches, num_classes, num_head, num_layer, T):
        super().__init__()
        self.T = T
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_dims, dims, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dims),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dims, dims, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(dims),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dims, dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dims),
            nn.LeakyReLU(0.1),
        )
        self.linears = nn.Sequential(
            nn.Linear(dims, dims),
            nn.BatchNorm1d(dims),
            nn.LeakyReLU(0.1),
            nn.Linear(dims, num_classes)
        )
        self.ap = nn.AdaptiveAvgPool2d(1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B, C, D = x.shape
        H = int(D ** 0.5)
        x = x.view(-1, self.T, C, H, H)
        x = x.mean(1)
        x = self.proj_conv(x)
        x = self.ap(x)
        x = x.flatten(1)
        x = self.linears(x)
        return x

if __name__ == '__main__':
    embd_dims = 384
    num_classes = 100
    num_patches = 256
    T = 2
    model = SurrogateModule(embd_dims, embd_dims, num_patches, num_classes, 8, 1, T)
    x = torch.randn(4, 384, 256)
    y = model(x)
    print(y.shape)