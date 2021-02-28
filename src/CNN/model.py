class SegModel:
    def __init__(self, kwargs):
        """ Inits the Net arguments """
        for key, value in kwargs.items():
            self.__dict__.update(kwargs)
        self.model=None

    def save_model(self):
        self.model.save(self.save_dir+"/"+self.name)

    def train(self):
        self.model.compile(
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy']
                )
                      
        #Create Generator
        if self.validate:
            self.history=self.model.fit(
                self.data._loaded_data,
                #validation_data=(self.Xtest, self.ytest),
                callbacks=self.callbacks,
            )
        else:
            pass


    def predict(X,y):
        pass