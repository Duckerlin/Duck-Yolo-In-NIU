# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os
    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""
from .BiFPN import *
from .block import (
    C1,
    C2,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    C3STR,
    SPPCSPC,
    SimConv,
    SimSPPF
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
    VoVGSCSP,       
    VoVGSCSPC,       
    GSConv,          
    ECA,
    space_to_depth,
    Focus,
    AKConv,


)
from .attention import (GAM)
from .shufflenet import ShuffleNetV2 , Conv_maxpool
from .c3c2 import C3c2
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .swin_transformer import (SwinTransformer) 
from .BiFormer import (BiFormerBlock, C2f_BiLevelRoutingAttention, C3_BiLevelRoutingAttention)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv"
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "v10Detect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "LightConv",
    "RepConv",
    "ShuffleNetV2", 
    "Conv_maxpool",
    "SwinTransformer",
    "C2f_OD",
    "C3c2",
    "C3STR",
    "VoVGSCSP", 
    "VoVGSCSPC", 
    "GSConv",
    "BiFormerBlock",
    "C2f_BiLevelRoutingAttention",
    "C3_BiLevelRoutingAttention",
    "ECA",
    "mn_conv",
    "space_to_depth",
    "Focus",
    "SPPCSPC",
    "AKConv",
    "SimConv",
    "SimSPPF",
    "GAM"
)
