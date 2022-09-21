import timm
from torchvision import models 
from nets import mobilenet, resnet, vgg, mlp, densenet, seresnet, googlenet


def get_network(net_name, input_dim=None, num_classes=None, pretrained=True):    
    # available model list: https://pytorch.org/vision/stable/models.html
    if net_name == 'densenet201':
        model = densenet.densenet201(num_class=num_classes)
    elif net_name == 'densenet121':
        model = densenet.densenet121(num_class=num_classes)
    elif net_name == 'googlenet':
        model = googlenet.googlenet(num_class=num_classes)
    elif net_name == 'mobilenet':
        model = mobilenet.mobilenet(num_class=num_classes)
    elif net_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained, num_class=num_classes)
    elif net_name == 'resnet18':
        model = resnet.resnet18(num_class=num_classes)
    elif net_name == 'resnet34':
        model = resnet.resnet34(num_class=num_classes)
    elif net_name == 'vgg16':
        model = vgg.vgg16_bn()
    elif net_name == 'vgg19':
        model = vgg.vgg19_bn(num_class=num_classes)
    elif net_name == 'seresnet50':
        model = seresnet.seresnet50()
    elif net_name == 'seresnet152':
        model = seresnet.seresnet152()
    elif net_name == 'mlp':
        model = mlp.MLP(input_dim, num_classes)
    elif net_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    elif net_name == 'inception4':
        model = timm.create_model('inception_v4', pretrained=pretrained)
    elif net_name == 'yolo3':
        model = timm.create_model('volo_d3_224', pretrained=True)

    return model 
    