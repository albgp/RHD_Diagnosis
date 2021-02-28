from keras_segmentation.models.pspnet import vgg_pspnet as pspnet
from .model import SegModel

class PSPNet(SegModel):
    def __init__(self, kwargs):
        """ Inits the PSPNet arguments """
        super().__init__(kwargs)
        self.model = pspnet(n_classes=4,  input_height=384, input_width=384 )
        self.model.summary()