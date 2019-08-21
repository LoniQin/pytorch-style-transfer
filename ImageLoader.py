import PIL.Image as Image
from utils import *

class ImageLoader(object):

    def __init__(self, file_path, loader, device = None):
        self.device = device
        self.loader = loader
        if device == None:
            self.device = get_current_device()
        self.image = Image.open(file_path)
        self.image = loader(self.image)
        print(self.image.size())
        self.image = self.image.unsqueeze(0)
        print(self.image.size())


