from __future__ import print_function
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from ImageLoader import ImageLoader
from Normalization import Normalization
from ContentLoss import ContentLoss
from StyleLoss import StyleLoss
import copy
class StyleTransfer(object):
    def __init__(self,
                 size,
                 style_path,
                 content_path,
                 save_path = None,
                 device = None,
                 content_layers = ['conv_4'],
                 style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
                 step_count = 300,
                 style_weight = 1000000,
                 content_weight = 1,
                 transform_model = None):
        self.device = device
        if self.device == None:
            self.device = get_current_device()
        self.size = size
        self.style_path = style_path
        self.content_path = content_path
        self.loader = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.step_count = step_count
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.save_path = save_path
        self.step = 0
        self.transform_model = transform_model
        if self.transform_model == None:
            self.transform_model = models.vgg19(pretrained = True).features.to(self.device).eval()

    def load_image(self):
        self.style_image = ImageLoader(file_path = self.style_path, loader = self.loader, device = self.device)
        self.content_image = ImageLoader(file_path = self.content_path, loader = self.loader, device = self.device)
        assert self.style_image.image.size() == self.content_image.image.size(), "Style image and content image should have the same size"
        self.input_image = self.content_image.image.clone()

    def show_image(self, tensor, title = None):
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()
        print("Before squeeze:", image.size())
        image = image.squeeze(0)
        print("After squeeze:", image.size())
        image = unloader(image)
        print("After unload: ", image.size)
        #plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(3)

    def show_images(self):
        plt.figure()

        self.show_image(tensor = self.style_image.image, title = "Style Image")
        plt.figure()
        self.show_image(tensor = self.content_image.image, title = "Content Image")
        plt.figure()
        self.show_image(self.input_image, "Input Image")
        plt.show()

    def run(self):

        torch.autograd.set_detect_anomaly(True)
        normalization = Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(self.device)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)
        i = 0
        transform_model = copy.deepcopy(self.transform_model)
        for layer in transform_model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.name))
            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(self.content_image.image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
            if name in self.style_layers:
                target = model(self.style_image.image).detach()
                style_loss = StyleLoss(target)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i+1)]

        optimizer = optim.LBFGS([self.input_image.requires_grad_()])
        print('Optimizing..')
        self.step = 0
        while self.step <= self.step_count:
            def closure():
                self.input_image.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(self.input_image)
                style_score = 0
                content_score = 0
                for item in style_losses:
                    style_score += item.loss
                for item in content_losses:
                    content_score += item.loss
                style_score *= self.style_weight
                content_score *= self.content_weight
                loss = style_score + content_score
                loss.backward()
                self.step += 1
                #print("run {}".format(self.step))
                if self.step % 50 == 0:
                    print("Style loss: {:4f} Content loss: {:4f}".format(style_score.item(), content_score.item()))
                return content_score + content_score
            optimizer.step(closure)
        self.input_image.data.clamp_(0, 1)

        if self.save_path != None:
            unloader = transforms.ToPILImage()
            image = self.input_image.cpu().clone()
            image = image.squeeze(0)
            image = unloader(image)
            image.save(self.save_path)
            print("Image saved to ", self.save_path)





if __name__ == "__main__":
    root_dir = "./data/"
    transfer = StyleTransfer(size = (320, 500),
                             content_path = root_dir + "content.jpg",
                             style_path = root_dir + "style.jpg",
                             step_count=100,
                             save_path= root_dir + "output.jpg")
    transfer.load_image()
    transfer.run()
