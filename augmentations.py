# #####################################################################################################################
'''

This module provides various transformations their helper functions to apply to your dataset as augmentations during 
training.

TODO: remove the writing and reading from dummy image files for debugging, to make the transformations faster

'''
# #####################################################################################################################

# image processing
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.feature_extraction import image
import cv2
import torchvision.transforms.functional as TF
from PIL import Image

# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

import numpy 
import os


def getMeanStddev(loader):
    """ Find the mean and stddev for the images, to be used later in further transforms
        this can take a while to run, so we have the function quit, and you can copy its results into a config manually

    Args:
        loader: A DataLoader object containing your image dataset. The code below will just use the first index of 
                whatever your loader returns. (So it can handle filenames of images if you want to use them.)
    """
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        data = data[0]  
        batch_samples = data.size(0)
        print(batch_samples)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    
    print('mean ', mean)
    print('std ', std)
    quit()

class RandomGaussianBlur:
    """ some slides might be sharper than others; try to blur the sharp ones sometimes? """
    def __init__(self, multichannel=True):
        self.multichannel = multichannel

    def __call__(self, x):
        openCVim = numpy.array(x)
        openCVim = cv2.GaussianBlur(openCVim,(5,5),0)
        return Image.fromarray(openCVim)

class ApplyThreshold:
    """ try to binarize the image in various ways (most useful if we were doing cell segmentation) """
    def __init__(self, typeThreshold='binary', adaptive=False, multichannel=True, cutoff=5):
        self.multichannel = multichannel
        self.typeThreshold = typeThreshold
        self.adaptive = adaptive
        self.cutoff = cutoff

    def __call__(self, x):
        open_cv_image = numpy.array(x) 

        # convert to B&W (TODO: there is probably a more elegant way to do this -- want code to work regardless if 
        # original was greyscale or not)
        Image.fromarray(open_cv_image).save('./temp.tif')
        open_cv_image = cv2.imread('./temp.tif', cv2.IMREAD_GRAYSCALE)

        if self.adaptive:
            if self.typeThreshold == 'binary':
                open_cv_image = cv2.adaptiveThreshold(open_cv_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,7,self.cutoff)
            elif self.typeThreshold == 'mean':
                open_cv_image = cv2.adaptiveThreshold(open_cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,7,self.cutoff)
            # TODO: why is this broken for otsu?
            #elif self.typeThreshold == 'otsu':
            #    open_cv_image = cv2.adaptiveThreshold(open_cv_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #       cv2.THRESH_BINARY+cv2.THRESH_OTSU,7,5)

        else:
            if self.typeThreshold == 'binary':
                _,open_cv_image = cv2.threshold(open_cv_image,self.cutoff,255,cv2.THRESH_BINARY)
            elif self.typeThreshold == 'tozero':
                _,open_cv_image = cv2.threshold(open_cv_image,self.cutoff,255,cv2.THRESH_TOZERO)
            elif self.typeThreshold == 'otsu':
                _,open_cv_image = cv2.threshold(open_cv_image,self.cutoff,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        return Image.fromarray(open_cv_image).convert('RGB')

class ErosionAndDilation:
    """ this is the same as Openingl once again, trying to get the "interesting" parts of an image seperated from "debris"
        probably more useful if, again, we were trying to do some kind of segmentation """
    def __init__(self, multichannel=True):
        self.multichannel = multichannel

    def __call__(self, x):
        openCVim = numpy.array(x)
        kernel = numpy.ones((5,5),numpy.uint8)
        if openCVim.shape[-1] > 3:
            cv2.imwrite('dummy4.jpg', openCVim, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            openCVim = numpy.array(Image.open("./dummy4.jpg"))

        openCVim = cv2.morphologyEx(openCVim, cv2.MORPH_OPEN, kernel)
        return Image.fromarray(openCVim)

class TopHat:
    """ an intelligent way to emphasize finer details in an image when trying to binarize (which it could/should be 
        combined with) """
    def __init__(self, multichannel=True, kernel=10):
        self.multichannel = multichannel
        self.kernel = kernel

    def __call__(self, x):
        openCVim = numpy.array(x)
        kernel = numpy.ones((self.kernel,self.kernel),numpy.uint8)
        if openCVim.shape[-1] > 3:
            cv2.imwrite('dummy4.jpg', openCVim, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            openCVim = numpy.array(Image.open("./dummy4.jpg"))

        openCVim = cv2.morphologyEx(openCVim, cv2.MORPH_TOPHAT, kernel)
        return Image.fromarray(openCVim)

class SobelX:
    """ detects and emphasizes edges """
    def __init__(self, multichannel=True):
        self.multichannel = multichannel

    def __call__(self, x):
        openCVim = numpy.array(x)
        if openCVim.shape[-1] > 3:
            cv2.imwrite('dummy4.jpg', openCVim, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            openCVim = numpy.array(Image.open("./dummy4.jpg"))
        openCVim = cv2.Sobel(openCVim,cv2.CV_64F,1,0,ksize=5)
        return Image.fromarray((openCVim * 255).astype(numpy.uint8))

class ContrastThroughHistogram:
    """ increases the contrast in an image through a histogram """
    def __init__(self, multichannel=True):
        self.multichannel = multichannel

    def __call__(self, x):
        openCVim = numpy.array(x)
        if openCVim.shape[-1] > 3:
            cv2.imwrite('dummy4.jpg', openCVim, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            openCVim = numpy.array(Image.open("./dummy4.jpg"))

        Image.fromarray(openCVim).save('./temp.tif')
        openCVim = cv2.imread('./temp.tif', cv2.IMREAD_GRAYSCALE)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        openCVim = clahe.apply(openCVim)

        Image.fromarray(openCVim).save('./temp.tif')
        openCVim = cv2.imread('./temp.tif')
        return Image.fromarray(openCVim)

class LocalBinaryPattern:
    """ converts the image to an array of integers, one per pixel, that are calculated using local histograms of 
        brightness of pixels of some radius """
    def __init__(self, multichannel=True):
        self.multichannel = multichannel

    def __call__(self, x):
        openCVim = numpy.array(x)
        radius = 10
        n_points = 9 * radius
        METHOD = 'uniform'

        if os.path.exists('./temp.tif'):
            os.system('rm ./temp.tif')
        Image.fromarray(openCVim).save('./temp.tif')
        openCVim = cv2.imread('./temp.tif', cv2.IMREAD_GRAYSCALE)
        openCVim = local_binary_pattern(openCVim, n_points, radius, METHOD)
        return Image.fromarray(openCVim).convert('RGB')

class Grayscale:
    """ converts an RBG image to greyscale, and then back to RBG to be compatible with various transforms and/or 
        transfer learning (which requires RBG) """
    def __init__(self, multichannel=True):
        self.multichannel = multichannel

    def __call__(self, x):
        x = x.convert('L')
        return x

class RGB:
    """ converts an RBG image to greyscale, and then back to RBG to be compatible with various transforms and/or 
        transfer learning (which requires RBG) """
    def __init__(self, multichannel=True):
        self.multichannel = multichannel

    def __call__(self, x):
        openCVim = numpy.array(x)
        Image.fromarray(openCVim).save('./temp.tif')
        openCVim = cv2.imread('./temp.tif', cv2.IMREAD_GRAYSCALE)
        return Image.fromarray(openCVim).convert('RGB')

class Debug:
    """ saves image for manual examination """
    def __init__(self, multichannel=True):
        self.multichannel = multichannel

    def __call__(self, x):
        print("in debug ", type(x))
        print(x)
        print(list(numpy.asarray(x)))
        x.save("./final.jpeg", "JPEG")
        return x

