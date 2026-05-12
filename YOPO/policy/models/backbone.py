import time
import torch
import torch.nn
from policy.models.resnet import resnet18


class RangeImageBackbone(torch.nn.Module):
    def __init__(self, output_dim: int, in_channels: int = 5):
        super(RangeImageBackbone, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.output_layer = torch.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, range_image: torch.Tensor) -> torch.Tensor:
        return self.cnn(range_image)


def YopoBackbone(output_dim, in_channels=5):
    return RangeImageBackbone(output_dim, in_channels=in_channels)


if __name__ == '__main__':
    net = YopoBackbone(64, in_channels=5)
    input_ = torch.zeros((1, 5, 96, 640))
    start = time.time()
    output = net(input_)
    print(f"Output shape: {output.shape}, Time: {time.time() - start:.4f}s")
