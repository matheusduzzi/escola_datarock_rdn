"""Baixa os pesos da ResNet-18 pronta (ImageNet) do endereço oficial do PyTorch e confere o SHA-256.

    python3 baixar_pesos.py

São 44,7 MB, uma vez só, para pesos/resnet18-f37072fd.pth. O torchvision usa esse mesmo arquivo:
o pedaço f37072fd do nome é o começo do SHA-256 dele. Se o SHA-256 não bater, o arquivo é apagado.
Precisa de internet só nessa hora. Não precisa do PyTorch.
"""
import resnet

print(resnet.baixar_pesos())
