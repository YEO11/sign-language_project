import torch.nn as nn
from types import SimpleNamespace
from timesformer.models.vit import vit_base_patch16_224
from timesformer.utils.parser import load_config

class TimeSformerClassifier(nn.Module):
    def __init__(self, num_classes=100, cfg_path="TimeSformer/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"):
        super().__init__()

        #  필요한 속성 모두 넘김
        cfg = load_config(SimpleNamespace(cfg_file=cfg_path, opts=[]))

        cfg.MODEL.NUM_CLASSES = num_classes
        self.backbone = vit_base_patch16_224(cfg)

    def forward(self, x):  # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        return self.backbone(x)
