
'''

This file is a configuration for the models tested in our fourth blog post; comparing using CellNET to ImageNet transfer 
learning.

Link to blog post: FIXME

'''

import copy
import os
from torchvision import transforms
from augmentations import *

# #####################################################################################################################

resnet_only__malaria_cell_images_baseline_mean = [0.5295, 0.4239, 0.4530]
resnet_only__malaria_cell_images_baseline_std = [0.3366, 0.2723, 0.2876]

train_transforms_380_greyscale_blog3 = [
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(),
        #RandomGaussianBlur(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Grayscale(),
        transforms.Resize(size=[380,380]),
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, 
                resnet_only__malaria_cell_images_baseline_std)]

eval_transforms_380_greyscale_blog3 = [
        Grayscale(),
        transforms.Resize(size=[380,380]),
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, 
                resnet_only__malaria_cell_images_baseline_std)
        ]

train_transforms_224_RGB_blog3 = [
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(),
        #RandomGaussianBlur(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(size=[224,224]),
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, 
                resnet_only__malaria_cell_images_baseline_std)]

eval_transforms_224_RGB_blog3 = [
        transforms.Resize(size=[224,224]),
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, 
                resnet_only__malaria_cell_images_baseline_std)
        ]

# #####################################################################################################################

vgg_only__kaggle_baseline_mean = [0.0918, 0.0918, 0.0918]
vgg_only__kaggle_baseline_std = [0.1675, 0.1675, 0.1675]

vgg_only__kaggle_blog3 = {}
vgg_only__kaggle_blog3['traindir'] = 'kaggle_bowl'
vgg_only__kaggle_blog3['testdir'] = 'kaggle_bowl'
vgg_only__kaggle_blog3['labels'] = os.listdir(vgg_only__kaggle_blog3['traindir'])
vgg_only__kaggle_blog3['max_class_samples'] = None #400
vgg_only__kaggle_blog3['usesCV'] = True

vgg_1channel_covid19_noImageNet__kaggle_blog3 = copy.deepcopy(vgg_only__kaggle_blog3)
vgg_1channel_covid19_noImageNet__kaggle_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withoutImageNet50000samples_.torch", n_classes=8, learning_rate=1e-3)'
vgg_1channel_covid19_noImageNet__kaggle_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_noImageNet__kaggle_blog3['traindir'] + "_" + "withoutImageNet"
vgg_1channel_covid19_noImageNet__kaggle_blog3['train_transforms'] = train_transforms_380_greyscale_blog3
vgg_1channel_covid19_noImageNet__kaggle_blog3['eval_transforms'] = eval_transforms_380_greyscale_blog3
vgg_1channel_covid19_noImageNet__kaggle_blog3['train_transforms'][-1] = transforms.Normalize([0.0918], [0.1675])
vgg_1channel_covid19_noImageNet__kaggle_blog3['eval_transforms'][-1] = transforms.Normalize([0.0918], [0.1675])

vgg_1channel_covid19_withImageNet__kaggle_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__kaggle_blog3)
vgg_1channel_covid19_withImageNet__kaggle_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withImageNet50000samples_.torch", n_classes=8, learning_rate=1e-3)'
vgg_1channel_covid19_withImageNet__kaggle_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_withImageNet__kaggle_blog3['traindir'] + "_" + "withImageNet"

vgg_1channel_withImageNet_only__kaggle_blog3 = copy.deepcopy(vgg_1channel_covid19_withImageNet__kaggle_blog3)
vgg_1channel_withImageNet_only__kaggle_blog3['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=8, learning_rate=1e-3, pretrained=True)'
vgg_1channel_withImageNet_only__kaggle_blog3['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_withImageNet_only__kaggle_blog3['traindir'] + "_" + "withImageNet"

vgg_1channel_noImageNet_only__kaggle_blog3 = copy.deepcopy(vgg_1channel_covid19_withImageNet__kaggle_blog3)
vgg_1channel_noImageNet_only__kaggle_blog3['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=8, learning_rate=1e-3, pretrained=False)'
vgg_1channel_noImageNet_only__kaggle_blog3['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_noImageNet_only__kaggle_blog3['traindir'] + "_" + "withoutImageNet"

vgg_3channel_covid19_noImageNet__kaggle_blog3 = copy.deepcopy(vgg_1channel_noImageNet_only__kaggle_blog3)
vgg_3channel_covid19_noImageNet__kaggle_blog3['model'] = 'Vgg19ThreeChannelModelAllLayersCovid19(n_classes=8, learning_rate=1e-3, saved_model="./cellnet_Vgg19_3channel_rotations_covid_withoutImageNet50000samples_.torch")'
vgg_3channel_covid19_noImageNet__kaggle_blog3['name'] = 'Vgg19ThreeChannelModelAllLayersCovid19' + "_" + vgg_3channel_covid19_noImageNet__kaggle_blog3['traindir'] + "_" + "withoutImageNet"
vgg_3channel_covid19_noImageNet__kaggle_blog3['train_transforms'] = train_transforms_224_RGB_blog3
vgg_3channel_covid19_noImageNet__kaggle_blog3['eval_transforms'] = eval_transforms_224_RGB_blog3
vgg_3channel_covid19_noImageNet__kaggle_blog3['train_transforms'][-1] = transforms.Normalize([0.0918], [0.1675])
vgg_3channel_covid19_noImageNet__kaggle_blog3['eval_transforms'][-1] = transforms.Normalize([0.0918], [0.1675])

vgg_3channel_covid19_withImageNet__kaggle_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__kaggle_blog3)
vgg_3channel_covid19_withImageNet__kaggle_blog3['model'] = 'Vgg19ThreeChannelModelAllLayersCovid19(n_classes=8, learning_rate=1e-3, saved_model="./cellnet_Vgg19_3channel_rotations_covid_withImageNet50000samples_.torch")'
vgg_3channel_covid19_withImageNet__kaggle_blog3['name'] = 'Vgg19ThreeChannelModelAllLayersCovid19' + "_" + vgg_3channel_covid19_withImageNet__kaggle_blog3['traindir'] + "_" + "withImageNet"

vgg_3channel_withImageNet_only__kaggle_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__kaggle_blog3)
vgg_3channel_withImageNet_only__kaggle_blog3['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=8, learning_rate=1e-3, pretrained=True)'
vgg_3channel_withImageNet_only__kaggle_blog3['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg_3channel_withImageNet_only__kaggle_blog3['traindir'] + "_" + "withImageNet"

vgg_3channel_noImageNet_only__kaggle_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__kaggle_blog3)
vgg_3channel_noImageNet_only__kaggle_blog3['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=8, learning_rate=1e-3, pretrained=False)'
vgg_3channel_noImageNet_only__kaggle_blog3['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg_3channel_noImageNet_only__kaggle_blog3['traindir'] + "_" + "withoutImageNet"

cnn_grey_covid19_only__kaggle_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__kaggle_blog3)
cnn_grey_covid19_only__kaggle_blog3['model'] = 'CNNGreyModelCovid19(n_classes=8, learning_rate=1e-3, saved_model="./cellnet_CNNGrey_1channel_rotations_covid_50000samples_.torch")'
cnn_grey_covid19_only__kaggle_blog3['name'] = 'CNNGreyModelCovid19' + "_" + cnn_grey_covid19_only__kaggle_blog3['traindir'] + "_" 



# #########################################################################################################################################################


