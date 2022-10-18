import torchvision
from models import IntermediateLayerGetter
import torch

model = torchvision.models.resnet18()
print('==============Origin Model==================')
print(model)
print('============================================')

return_layers = {'layer3': 'feature_3', 'layer1': 'feature_1'}
backbone = IntermediateLayerGetter(model, return_layers)
print('==============Backbone Model==================')
print(backbone)
print('==============================================')
backbone.eval()
x = torch.randn(1, 3, 224, 224)
out = backbone(x)

print('==============Output==================')
print(out)
print('==============================================')

print(out['feature_3'].shape, out['feature_1'].shape)
