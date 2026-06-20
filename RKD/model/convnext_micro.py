import torch
import torch.nn as nn

from .convnext_block import ConvNextBlock, Downsample

__all__ = ['ConvNextMicro']


class ConvNextMicro(nn.Module):
    """Small 4-stage ConvNeXt.

    Configurable via ``dims`` (channels per stage) and ``depths`` (number of
    ConvNextBlocks per stage). The default is a lightweight variant with
    < 1M parameters: dims [24, 48, 96, 192], depths [1, 1, 3, 1].

    Spatial / channel flow for a 224x224x3 input (default dims):

        input                                224 x 224 x 3
        Conv2d 4x4, stride 4 (stem)           56 x  56 x 24
        ConvNextBlock (24)                     56 x  56 x 24
        Downsample (24  -> 48)                 28 x  28 x 48
        ConvNextBlock (48)                     28 x  28 x 48
        Downsample (48  -> 96)                14 x  14 x 96
        ConvNextBlock (96) x3                 14 x  14 x 96
        Downsample (96 -> 192)                 7 x   7 x 192
        ConvNextBlock (192)                    7 x   7 x 192
        Global average pooling                         192
        LayerNorm                                      192
        Linear                                 num_classes
        Softmax                                num_classes

    ``apply_softmax`` controls the final activation: keep it ``True`` to match
    the architecture spec, set it to ``False`` for training with
    ``nn.CrossEntropyLoss`` (which expects raw logits).

    ``drop_path`` is the *maximum* stochastic-depth rate; per-block rates are
    spread linearly from 0 to ``drop_path`` across all blocks (ConvNeXt recipe).
    """

    def __init__(self, num_classes=1000, in_channels=3, drop_path=0.0,
                 dims=(24, 48, 96, 192), depths=(1, 1, 3, 1),
                 apply_softmax=True, init_std=0.2):
        super(ConvNextMicro, self).__init__()

        assert len(dims) == 4 and len(depths) == 4
        self.init_std = init_std
        self.apply_softmax = apply_softmax

        # Linearly increasing stochastic-depth rate across all blocks.
        total_blocks = sum(depths)
        dp_rates = [r.item() for r in torch.linspace(0, drop_path, total_blocks)] \
            if total_blocks > 0 else []
        self._cursor = 0

        def stage(dim, n):
            blocks = [ConvNextBlock(dim, drop_path=dp_rates[self._cursor + i])
                      for i in range(n)]
            self._cursor += n
            return nn.Sequential(*blocks)

        # Stem: patchify the image with a 4x4 stride-4 conv (224 -> 56).
        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4)

        # Stage 1 @ 56x56
        self.stage1 = stage(dims[0], depths[0])

        # Stage 2 @ 28x28
        self.down1 = Downsample(dims[0], dims[1])
        self.stage2 = stage(dims[1], depths[1])

        # Stage 3 @ 14x14
        self.down2 = Downsample(dims[1], dims[2])
        self.stage3 = stage(dims[2], depths[2])

        # Stage 4 @ 7x7
        self.down3 = Downsample(dims[2], dims[3])
        self.stage4 = stage(dims[3], depths[3])

        # Head
        self.norm = nn.LayerNorm(dims[3], eps=1e-6)
        self.fc = nn.Linear(dims[3], num_classes)
        self.softmax = nn.Softmax(dim=1)

        del self._cursor
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # ConvNeXt recipe: truncated-normal weights, zero biases.
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """Return per-stage feature maps and the pooled, normalized embedding.

        Keys ``stage1..stage4`` are the (N, C, H, W) outputs of each stage
        (stage2 is the AT attachment point used for distillation); ``embedding``
        is the (N, dims[-1]) vector after global average pooling + LayerNorm,
        i.e. the classifier's input. Resolutions for a 224 input:
        stage1 56x56, stage2 28x28, stage3 14x14, stage4 7x7.
        """
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(self.down1(s1))
        s3 = self.stage3(self.down2(s2))
        s4 = self.stage4(self.down3(s3))

        # Global average pooling over H, W -> (N, C), then LayerNorm.
        emb = self.norm(s4.mean(dim=[-2, -1]))
        return {"stage1": s1, "stage2": s2, "stage3": s3, "stage4": s4,
                "embedding": emb}

    def forward(self, x):
        x = self.fc(self.forward_features(x)["embedding"])
        if self.apply_softmax:
            return self.softmax(x)
        return x
