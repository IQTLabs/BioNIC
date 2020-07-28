
'''

This file is a configuration for the models tested in our fourth blog post; comparing using CellNET to ImageNet transfer learning

'''

import copy
import os
from torchvision import transforms
from augmentations import *

# #########################################################################################################################################################
# kaggle_bowl
# #########################################################################################################################################################

# #########################################################################################################################################################
resnet_only__malaria_cell_images_baseline_mean = [0.5295, 0.4239, 0.4530]
resnet_only__malaria_cell_images_baseline_std = [0.3366, 0.2723, 0.2876]
train_transforms_380_greyscale_blog1 = [
                #RGB(),
                transforms.RandomRotation(degrees=90),
                transforms.ColorJitter(),
                RandomGaussianBlur(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Grayscale(),
                transforms.Resize(size=[380,380]),
                #TopHat(),
                #ApplyThreshold(typeThreshold='otsu', adaptive=False),
                transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, resnet_only__malaria_cell_images_baseline_std)]

eval_transforms_380_greyscale_blog1 = [
        Grayscale(),
        transforms.Resize(size=[380,380]),
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, resnet_only__malaria_cell_images_baseline_std)
        ]


# #########################################################################################################################################################

vgg_only__kaggle_baseline_mean = [0.0918, 0.0918, 0.0918]
vgg_only__kaggle_baseline_std = [0.1675, 0.1675, 0.1675]

vgg_only__kaggle_blog1 = {}
vgg_only__kaggle_blog1['traindir'] = 'kaggle_bowl'
vgg_only__kaggle_blog1['testdir'] = 'kaggle_bowl'
vgg_only__kaggle_blog1['labels'] = os.listdir(vgg_only__kaggle_blog1['traindir'])
vgg_only__kaggle_blog1['max_class_samples'] = None #400
vgg_only__kaggle_blog1['usesCV'] = True

vgg_1channel_covid19_noImageNet__kaggle_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
vgg_1channel_covid19_noImageNet__kaggle_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_vgg19_1channel_rotations_covid_withoutImageNet_50Ksamples_.torch", n_classes=8, learning_rate=learning_rate)'
vgg_1channel_covid19_noImageNet__kaggle_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_noImageNet__kaggle_blog1['traindir'] + "_" + "blog1"
vgg_1channel_covid19_noImageNet__kaggle_blog1['train_transforms'] = train_transforms_380_greyscale_blog1
vgg_1channel_covid19_noImageNet__kaggle_blog1['eval_transforms'] = eval_transforms_380_greyscale_blog1
vgg_1channel_covid19_noImageNet__kaggle_blog1['train_transforms'][-1] = transforms.Normalize([0.0918], [0.1675])
vgg_1channel_covid19_noImageNet__kaggle_blog1['eval_transforms'][-1] = transforms.Normalize([0.0918], [0.1675])

vgg_1channel_covid19_withImageNet__kaggle_blog1 = copy.deepcopy(vgg_1channel_covid19_noImageNet__kaggle_blog1)
vgg_1channel_covid19_withImageNet__kaggle_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_vgg19_1channel_rotations_covid_withImageNet_50Ksamples_.torch", n_classes=8, learning_rate=learning_rate)'
vgg_1channel_covid19_withImageNet__kaggle_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_withImageNet__kaggle_blog1['traindir'] + "_" + "blog1"

vgg_1channel_withImageNet_only__kaggle_blog1 = copy.deepcopy(vgg_1channel_covid19_noImageNet__kaggle_blog1)
vgg_1channel_withImageNet_only__kaggle_blog1['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=8, learning_rate=learning_rate, pretrained=True)'
vgg_1channel_withImageNet_only__kaggle_blog1['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_withImageNet_only__kaggle_blog1['traindir'] + "_" + "blog1"

vgg_1channel_noImageNet_only__kaggle_blog1 = copy.deepcopy(vgg_1channel_covid19_noImageNet__kaggle_blog1)
vgg_1channel_noImageNet_only__kaggle_blog1['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=8, learning_rate=learning_rate, pretrained=False)'
vgg_1channel_noImageNet_only__kaggle_blog1['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_noImageNet_only__kaggle_blog1['traindir'] + "_" + "blog1"



# #########################################################################################################################################################


