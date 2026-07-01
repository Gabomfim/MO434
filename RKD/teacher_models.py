"""Backbones de professor para classificação, com interface uniforme.

Cada wrapper expõe:
  * ``forward(x) -> logits``
  * ``forward_features(x) -> {"stage2", "embedding", "logits"}``
  * ``.head`` (a camada Linear final, p/ learning rate diferenciado)

``stage2`` é sempre o mapa de ativação 28x28 do 2º estágio, pareável com o
stage-2 da ConvNextMicro (o AT agrega sobre canais, então a diferença de
canais não importa). ``embedding`` é o vetor agrupado pré-classificador.

O mesmo wrapper é usado no fine-tune (treina ``forward``) e na destilação
(usa ``forward_features``), então o checkpoint salvo no fine-tune é carregado
diretamente como professor (mesmo formato de ``state_dict``).
"""

import torch
import torch.nn as nn
import torchvision


class ResNet18Teacher(nn.Module):
    """ResNet-18: stage2 = saída de layer2 (28x28), embedding = 512."""
    embedding_dim = 512

    def __init__(self, num_classes, pretrained=False, freeze_backbone=False):
        super().__init__()
        weights = (torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                   if pretrained else None)
        self.m = torchvision.models.resnet18(weights=weights)
        if freeze_backbone:
            for p in self.m.parameters():
                p.requires_grad = False
        self.m.fc = nn.Linear(self.m.fc.in_features, num_classes)

    @property
    def head(self):
        return self.m.fc

    def forward_features(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        l1 = m.layer1(x)
        l2 = m.layer2(l1)              # 28x28 -> AT
        l4 = m.layer4(m.layer3(l2))
        emb = torch.flatten(m.avgpool(l4), 1)
        return {"stage2": l2, "embedding": emb, "logits": m.fc(emb)}

    def forward(self, x):
        return self.forward_features(x)["logits"]


class ConvNextTinyTeacher(nn.Module):
    """ConvNeXt-Tiny: stage2 = saída de features[:4] (28x28), embedding = 768."""
    embedding_dim = 768

    def __init__(self, num_classes, pretrained=False, freeze_backbone=False):
        super().__init__()
        weights = (torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
                   if pretrained else None)
        self.m = torchvision.models.convnext_tiny(weights=weights)
        if freeze_backbone:
            for p in self.m.parameters():
                p.requires_grad = False
        in_f = self.m.classifier[2].in_features
        self.m.classifier[2] = nn.Linear(in_f, num_classes)

    @property
    def head(self):
        return self.m.classifier[2]

    def forward_features(self, x):
        f = x
        stage2 = None
        for i, blk in enumerate(self.m.features):
            f = blk(f)
            if i == 3:                # após o 2º estágio (28x28)
                stage2 = f
        pooled = self.m.avgpool(f)              # (N, C, 1, 1)
        emb = torch.flatten(pooled, 1)          # (N, 768)
        logits = self.m.classifier(pooled)      # LayerNorm2d -> Flatten -> Linear
        return {"stage2": stage2, "embedding": emb, "logits": logits}

    def forward(self, x):
        return self.forward_features(x)["logits"]


# arch -> (classe, otimizador default p/ fine-tune, lr default do backbone)
ARCHS = {
    "resnet18": {"cls": ResNet18Teacher, "opt": "sgd", "lr": 0.01},
    "convnext_tiny": {"cls": ConvNextTinyTeacher, "opt": "adamw", "lr": 1e-4},
}


def build_classifier(arch, num_classes, pretrained=True, freeze_backbone=False):
    """Instancia o wrapper de classificador para `arch`."""
    return ARCHS[arch]["cls"](num_classes, pretrained=pretrained,
                              freeze_backbone=freeze_backbone)


def load_teacher(arch, num_classes, ckpt_path, device, strict=True):
    """Instancia o wrapper e carrega um checkpoint do fine-tune (chave 'model'
    ou state_dict cru).

    ``strict=False`` ignora divergências (ex.: cabeça classificadora não usada
    em metric learning), carregando apenas o backbone/embedding.
    """
    model = build_classifier(arch, num_classes, pretrained=False)
    blob = torch.load(ckpt_path, map_location=device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    if not strict:
        # strict=False do PyTorch ignora apenas chaves faltando/sobrando, NÃO
        # divergências de shape de uma chave presente nos dois lados. A cabeça
        # classificadora (m.fc / classifier[2]) muda de nº de classes entre
        # datasets e não é usada em metric learning, então descartamos qualquer
        # param com shape incompatível e carregamos só o backbone/embedding.
        msd = model.state_dict()
        state = {k: v for k, v in state.items()
                 if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(state, strict=strict)
    return model.to(device)
