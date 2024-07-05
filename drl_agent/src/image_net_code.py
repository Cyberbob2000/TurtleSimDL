import torch
import torchvision.models
model_ft = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
### strip the last layer
feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
### check this works
x = torch.randn([1,3,224,224])
output = feature_extractor(x) # output now has the features corresponding to input x
print(output.shape)
print(output)