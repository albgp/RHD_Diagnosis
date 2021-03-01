def preprocess_RSofia(img):
    """
        Simple and cheap function to remove text on both sides of an echocardiography in the 
        template for the available images 
    """
    for i in range(480):
        for j in range(200):
            img[i,j]=0
    for i in range(150):
        for j in range(500):
            img[i,j]=0 
    for i in range(475):
        for j in range(90):
            img[i,img.shape[1]-1-j]=0
    for i in range(90):
        for j in range(465):
            img[i,img.shape[1]-1-j]=0
    return img