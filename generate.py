import sys
import torch
from torch import nn
import torchvision.utils as vutils


# ---------------------------------------
# Generator Model
# ---------------------------------------

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            self.conv_block(100, 128, padding=0),
            self.conv_block(128, 64, stride=2, ks=3),
            self.conv_block(64, 32, stride=2),
            self.conv_block(32, 1, stride=2, bn=False, out_layer=True)
        )

    @staticmethod
    def conv_block(
        in_c,
        out_c,
        out_layer=False,
        ks=4,
        stride=1,
        padding=1,
        bias=False,
        bn=True
    ):
        layers = [
            nn.ConvTranspose2d(
                in_c,
                out_c,
                ks,
                stride=stride,
                padding=padding,
                bias=bias
            )
        ]

        if bn:
            layers.append(nn.BatchNorm2d(out_c))

        if out_layer:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------
# Load checkpoint
# ---------------------------------------

checkpoint_path = sys.argv[1]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

G = Generator().to(device)

G.load_state_dict(
    torch.load(checkpoint_path, map_location=device)
)

G.eval()

# ---------------------------------------
# Generate images
# ---------------------------------------

with torch.no_grad():
    noise = torch.randn((16, 100, 1, 1), device=device)
    fake_images = G(noise).cpu()

    vutils.save_image(
        fake_images,
        "generated_digits.png",
        normalize=True,
        nrow=4
    )

print("Generated images saved to generated_digits.png")