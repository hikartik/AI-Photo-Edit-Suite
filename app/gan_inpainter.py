import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Conv2d):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=bias)
        # one-channel mask kernel
        self.register_buffer("mask_kernel", torch.ones(1, 1, kernel_size, kernel_size))
        self.slide_winsize = float(self.mask_kernel.numel())

    def forward(self, x, mask):
        # x: B×C×H×W, mask: B×1×H×W
        valid = F.conv2d(mask, self.mask_kernel, 
                         stride=self.stride, padding=self.padding)
        mask_ratio   = self.slide_winsize / (valid + 1e-8)
        updated_mask = torch.clamp(valid, 0, 1)
        mask_ratio   = mask_ratio * updated_mask

        x_masked = x * mask
        raw_out  = super().forward(x_masked)

        if self.bias is not None:
            b = self.bias.view(1, -1, 1, 1)
            out = (raw_out - b) * mask_ratio + b
        else:
            out = raw_out * mask_ratio

        return out, updated_mask
    

# Your PartialConv2d from before (single-channel mask version)
class PConvUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: downsample via stride=2
        self.enc1 = PartialConv2d(3,   64, kernel_size=7, padding=3, stride=1)
        self.enc2 = PartialConv2d(64, 128, kernel_size=5, padding=2, stride=2)
        self.enc3 = PartialConv2d(128,256, kernel_size=5, padding=2, stride=2)
        self.enc4 = PartialConv2d(256,512, kernel_size=3, padding=1, stride=2)
        # Decoder: three upsampling steps
        self.dec4 = PartialConv2d(512+256,256, kernel_size=3, padding=1)
        self.dec3 = PartialConv2d(256+128,128, kernel_size=3, padding=1)
        self.dec2 = PartialConv2d(128+64, 64,  kernel_size=3, padding=1)
        # Final conv: combine with original RGB
        self.dec1 = nn.Conv2d(64+3, 3, kernel_size=7, padding=3)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        # Encoder
        x1, m1 = self.enc1(x, mask);    x1, m1 = self.relu(x1), m1
        x2, m2 = self.enc2(x1, m1);     x2, m2 = self.relu(x2), m2
        x3, m3 = self.enc3(x2, m2);     x3, m3 = self.relu(x3), m3
        x4, m4 = self.enc4(x3, m3);     x4, m4 = self.relu(x4), m4

        # Decoder step 1: 32→64
        d4  = F.interpolate(x4, scale_factor=2, mode="nearest")
        dm4 = F.interpolate(m4, scale_factor=2, mode="nearest")
        out4, m5 = self.dec4(torch.cat([d4, x3], dim=1), dm4 * m3)
        d4, m5   = self.relu(out4), m5

        # Decoder step 2: 64→128
        d3  = F.interpolate(d4, scale_factor=2, mode="nearest")
        dm5 = F.interpolate(m5, scale_factor=2, mode="nearest")
        out3, m6 = self.dec3(torch.cat([d3, x2], dim=1), dm5 * m2)
        d3, m6   = self.relu(out3), m6

        # Decoder step 3: 128→256
        d2  = F.interpolate(d3, scale_factor=2, mode="nearest")
        dm6 = F.interpolate(m6, scale_factor=2, mode="nearest")
        out2, m7 = self.dec2(torch.cat([d2, x1], dim=1), dm6 * m1)
        d2, m7   = self.relu(out2), m7

        # Final conv (no mask)
        out = self.dec1(torch.cat([d2, x], dim=1))
        return self.tanh(out), m7

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def down(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1)]
            if norm: layers.append(nn.InstanceNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            down(3, 64, norm=False),
            down(64,128),
            down(128,256),
            down(256,512),
            nn.Conv2d(512,1,4,1,1)
        )
    def forward(self, x):
        return self.net(x)
    
    
def load_generator(weights_path: str, device=None) -> PConvUNet:
    """
    Instantiate PConvUNet, load weights from weights_path, and return eval model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PConvUNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
