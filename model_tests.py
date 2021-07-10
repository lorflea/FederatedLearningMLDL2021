from flopth import flopth
import torchvision.models as models



if __name__ == '__main__':
  resnet18 = models.resnet18(pretrained=True)
  sum_flops = flopth(resnet18, in_size=[[3], [10]])
  print(sum_flops)