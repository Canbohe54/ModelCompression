import torch
import yaml
import torchsummary
from thop import profile
from ptflops import get_model_complexity_info
from util.torchstat import stat


def readconfig(config_path="config.yaml"):
    global input_model_path, output_model_path, img_size, img_channels, mat3D, mat4D
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        input_model_path = config["input_model_path"]
        output_model_path = config["output_model_path"]
        img_size = config["img_size"]
        img_channels = config["img_channels"]
    mat3D = (img_channels, img_size, img_size)
    mat4D = (1, img_channels, img_size, img_size)


def get_network_info(model):
    print('Torch-Summary'.center(32, '#'))
    torchsummary.summary(model, mat3D)

    print('thop'.center(32, '#'))
    dummy_input = torch.rand(mat4D)
    macs, params = profile(model.to('cpu'), inputs=(dummy_input,))

    print('macs: ', macs, 'params: ', params)
    print('macs: %.2f M, params: %.2f M' % (macs / 1000000.0, params / 1000000.0))

    print('ptflops'.center(32, '#'))
    macs, params = get_model_complexity_info(model, mat3D)
    print('Computational complexity(Multiply-Accumulates):', macs, ', params:', params)

    print('torchstat'.center(32, '#'))
    print(stat(model.to('cpu'), mat3D))


if __name__ == "__main__":
    readconfig()
    input_model = torch.load(input_model_path)
    print('get input model info'.center(32, '.'))
    get_network_info(input_model)
    output_model = torch.load(output_model_path)
    print('get output model info'.center(32, '.'))
    get_network_info(output_model)
