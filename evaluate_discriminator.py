import sys
import torch
from torch import nn
from torchvision import datasets, transforms

# ---------------------------------------
# Device
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------
# Discriminator
# ---------------------------------------

def disc_conv(in_c, out_c, ks=4, stride=2, padding=1, bn=True, out_layer=False):
    layers = [
        nn.Conv2d(
            in_c,
            out_c,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False
        )
    ]

    if bn:
        layers.append(nn.BatchNorm2d(out_c))

    layers.append(
        nn.Sigmoid() if out_layer else nn.LeakyReLU(0.2)
    )

    return nn.Sequential(*layers)


D = nn.Sequential(
    disc_conv(1, 32, bn=False),
    disc_conv(32, 64),
    disc_conv(64, 128, ks=3),
    disc_conv(128, 1, out_layer=True, bn=False, padding=0)
)

# ---------------------------------------
# Generator
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


G = Generator()

# ---------------------------------------
# Load checkpoints
# ---------------------------------------

if len(sys.argv) != 3:
    print(
        "Usage: python evaluate_discriminator.py "
        "<generator_checkpoint> <discriminator_checkpoint>"
    )
    sys.exit(1)

generator_checkpoint = sys.argv[1]
discriminator_checkpoint = sys.argv[2]

G.load_state_dict(
    torch.load(generator_checkpoint, map_location=device)
)

D.load_state_dict(
    torch.load(discriminator_checkpoint, map_location=device)
)

G.to(device)
D.to(device)

G.eval()
D.eval()

# ---------------------------------------
# Load real MNIST image
# ---------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = datasets.MNIST(
    "MNIST_data/",
    train=False,
    download=True,
    transform=transform
)

real_image, label = testset[0]

real_image = real_image.unsqueeze(0).to(device)

# ---------------------------------------
# Generate fake image
# ---------------------------------------

noise = torch.randn(1, 100, 1, 1, device=device)

with torch.no_grad():
    fake_image = G(noise)

# ---------------------------------------
# Evaluate
# ---------------------------------------

with torch.no_grad():
    real_score = D(real_image).view(-1).item()
    fake_score = D(fake_image).view(-1).item()

print("\nDiscriminator Evaluation")
print("-" * 30)
print(f"Real MNIST digit label : {label}")
print(f"Real image score       : {real_score:.4f}")
print(f"Fake image score       : {fake_score:.4f}")