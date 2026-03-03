from torchinfo import summary
from layer.fg_mscnet import FG_MSCNet
model = FG_MSCNet(num_classes=1, img_size=512)
summary(model, input_size=(1, 3, 512, 512))
