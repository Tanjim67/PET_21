
# Constant definitions
# LOG LEVELS
DEBUG = 4
INFO = 3
WARN = 2
ERROR = 1
OFF = 0
DEBUG_LEVEL = INFO


# Log levels
def debug(s):
    if DEBUG_LEVEL >= DEBUG:
        print('[DEBUG] ' + str(s))

def info(s):
    if DEBUG_LEVEL >= INFO:
        print('[INFO]  ' + str(s))

def warn(s):
    if DEBUG_LEVEL >= WARN:
        print('[WARN]  ' + str(s))

def error(s):
    if DEBUG_LEVEL >= ERROR:
        print('[ERROR] ' + str(s))



# Helper functions
def showdata():
    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig('dataset.png')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
