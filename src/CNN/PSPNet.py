from keras_segmentation.models.pspnet import vgg_pspnet as pspnet

class PSPNet:
    def __init__(self, **kwargs):
    """ Inits the PSPNet arguments """
        for key, value in kwargs.items():
            self.__dict__.update(kwargs)

    def create_model(self):
        self.model = pspnet(n_classes=4,  input_height=384, input_width=384 )
        self.model.summary()

    def save_model(self):
        self.model.save(self.save_dir+"/"+self.name)

    def train(X, y, Xtest=None, y_test=None):
        pass 

    def predict(X,y):
        pass