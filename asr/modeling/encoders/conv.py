import torch
import torch.nn as nn


class Conv2dEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Conv2dEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=output_dim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=2
            ),
            nn.ReLU(),
        )
        self.output = nn.Linear(
            output_dim * (((input_dim - 1) // 2 - 1) // 2), output_dim
        )

    def forward(self, xs, xlens):
        xs = xs.unsqueeze(1)  # (B, 1, L, input_dim)
        xs = self.conv(xs)
        bs, output_dim, length, dim = xs.size()
        xs = self.output(
            xs.transpose(1, 2).contiguous().view(bs, length, output_dim * dim)
        )
        xlens = ((xlens - 1) // 2 - 1) // 2
        return xs, xlens


# DEBUG
if __name__ == "__main__":
    xs = torch.rand((5, 110, 80))
    xlens = torch.randint(1, 110, (5,))

    print("xs:", xs.shape)
    print("xlens:", xlens)

    input_dim = 80
    output_dim = 512
    conv = Conv2dEncoder(input_dim, output_dim)

    eouts, elens = conv(xs, xlens)
    print("eouts:", eouts.shape)
    print(eouts[0, :5, :5])
    print("elens:", elens)

    import os
    import sys

    EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    sys.path.append(EMOASR_ROOT)

    from asr.models.model_utils import make_src_mask
    from asr.models.transformer import Conv2dSubsampler

    conv2 = Conv2dSubsampler(input_dim, output_dim, 0)
    xs_mask = make_src_mask(xlens)
    eouts, eouts_mask = conv2(xs, xs_mask)
    print("eouts:", eouts.shape)
    print(eouts[0, :5, :5])
    print("eouts_mask:", eouts_mask)
    elens = torch.sum(eouts_mask, dim=1)
    print("elens:", elens)
