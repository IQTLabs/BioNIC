
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
grey__malaria_cell_images_baseline_mean = [0.4588]
grey__malaria_cell_images_baseline_std = [0.2844]

train_transforms_380_greyscale_blog3 = [
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(),
        #RandomGaussianBlur(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        Grayscale(),
        transforms.Resize(size=[380,380]),
        transforms.ToTensor(),
        transforms.Normalize(grey__malaria_cell_images_baseline_mean, 
                grey__malaria_cell_images_baseline_std)]

eval_transforms_380_greyscale_blog3 = [
        Grayscale(),
        transforms.Resize(size=[380,380]),
        transforms.ToTensor(),
        transforms.Normalize(grey__malaria_cell_images_baseline_mean, 
                grey__malaria_cell_images_baseline_std)
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

kaggle_baseline_mean = [0.0918, 0.0918, 0.0918]
kaggle_baseline_std = [0.1675, 0.1675, 0.1675]

base__kaggle_blog3 = {}
base__kaggle_blog3['traindir'] = 'kaggle_bowl'
base__kaggle_blog3['testdir'] = 'kaggle_bowl'
base__kaggle_blog3['labels'] = os.listdir(base__kaggle_blog3['traindir'])
base__kaggle_blog3['max_class_samples'] = None #400
base__kaggle_blog3['usesCV'] = True

vgg_1channel_covid19_noImageNet__kaggle_blog3 = copy.deepcopy(base__kaggle_blog3)
vgg_1channel_covid19_noImageNet__kaggle_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withoutImageNet50000samples_.torch", n_classes=8, learning_rate=1e-3)'
vgg_1channel_covid19_noImageNet__kaggle_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_noImageNet__kaggle_blog3['traindir'] + "_" + "withoutImageNet"
vgg_1channel_covid19_noImageNet__kaggle_blog3['train_transforms'] = train_transforms_380_greyscale_blog3
vgg_1channel_covid19_noImageNet__kaggle_blog3['eval_transforms'] = eval_transforms_380_greyscale_blog3
vgg_1channel_covid19_noImageNet__kaggle_blog3['train_transforms'][-1] = transforms.Normalize([kaggle_baseline_mean[0]], [kaggle_baseline_std[0]])
vgg_1channel_covid19_noImageNet__kaggle_blog3['eval_transforms'][-1] = transforms.Normalize([kaggle_baseline_mean[0]], [kaggle_baseline_std[0]])

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
vgg_3channel_covid19_noImageNet__kaggle_blog3['train_transforms'][-1] = transforms.Normalize(kaggle_baseline_mean, kaggle_baseline_std)
vgg_3channel_covid19_noImageNet__kaggle_blog3['eval_transforms'][-1] = transforms.Normalize(kaggle_baseline_mean, kaggle_baseline_std)

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
cnn_grey_covid19_only__kaggle_blog3['model'] = 'CNNGreyModelCovid19(n_classes=8, learning_rate=1e-4, saved_model="./cellnet_CNNGrey_1channel_rotations_covid_50000samples_.torch")'
cnn_grey_covid19_only__kaggle_blog3['name'] = 'CNNGreyModelCovid19' + "_" + cnn_grey_covid19_only__kaggle_blog3['traindir'] + "_" 


# #########################################################################################################################################################

base__malaria_cell_images_blog3 = copy.deepcopy(base__kaggle_blog3)
base__malaria_cell_images_blog3['traindir'] = 'malaria_cell_images'
base__malaria_cell_images_blog3['testdir'] = 'malaria_cell_images'
base__malaria_cell_images_blog3['labels'] = os.listdir(base__malaria_cell_images_blog3['traindir'])
base__malaria_cell_images_blog3['train_transforms'] = train_transforms_380_greyscale_blog3
base__malaria_cell_images_blog3['eval_transforms'] = eval_transforms_380_greyscale_blog3
base__malaria_cell_images_blog3['max_class_samples'] = 500

vgg_1channel_covid19_noImageNet__malaria_cell_images_blog3 = copy.deepcopy(base__malaria_cell_images_blog3)
vgg_1channel_covid19_noImageNet__malaria_cell_images_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withoutImageNet50000samples_.torch", n_classes=2, learning_rate=1e-3)'
vgg_1channel_covid19_noImageNet__malaria_cell_images_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_noImageNet__malaria_cell_images_blog3['traindir'] + "_" + "withoutImageNet"

vgg_1channel_covid19_withImageNet__malaria_cell_images_blog3 = copy.deepcopy(base__malaria_cell_images_blog3)
vgg_1channel_covid19_withImageNet__malaria_cell_images_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withImageNet50000samples_.torch", n_classes=2, learning_rate=1e-3)'
vgg_1channel_covid19_withImageNet__malaria_cell_images_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_withImageNet__malaria_cell_images_blog3['traindir'] + "_" + "withImageNet"

vgg_1channel_withImageNet_only__malaria_cell_images_blog3 = copy.deepcopy(base__malaria_cell_images_blog3)
vgg_1channel_withImageNet_only__malaria_cell_images_blog3['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=2, learning_rate=1e-3, pretrained=True)'
vgg_1channel_withImageNet_only__malaria_cell_images_blog3['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_withImageNet_only__malaria_cell_images_blog3['traindir'] + "_" + "withImageNet"

vgg_1channel_noImageNet_only__malaria_cell_images_blog3 = copy.deepcopy(base__malaria_cell_images_blog3)
vgg_1channel_noImageNet_only__malaria_cell_images_blog3['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=2, learning_rate=1e-3, pretrained=False)'
vgg_1channel_noImageNet_only__malaria_cell_images_blog3['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_noImageNet_only__malaria_cell_images_blog3['traindir'] + "_" + "withoutImageNet"

vgg_3channel_covid19_noImageNet__malaria_cell_images_blog3 = copy.deepcopy(vgg_1channel_noImageNet_only__malaria_cell_images_blog3)
vgg_3channel_covid19_noImageNet__malaria_cell_images_blog3['model'] = 'Vgg19ThreeChannelModelAllLayersCovid19(n_classes=2, learning_rate=1e-3, saved_model="./cellnet_Vgg19_3channel_rotations_covid_withoutImageNet50000samples_.torch")'
vgg_3channel_covid19_noImageNet__malaria_cell_images_blog3['name'] = 'Vgg19ThreeChannelModelAllLayersCovid19' + "_" + vgg_3channel_covid19_noImageNet__malaria_cell_images_blog3['traindir'] + "_" + "withoutImageNet"
vgg_3channel_covid19_noImageNet__malaria_cell_images_blog3['train_transforms'] = train_transforms_224_RGB_blog3
vgg_3channel_covid19_noImageNet__malaria_cell_images_blog3['eval_transforms'] = eval_transforms_224_RGB_blog3

vgg_3channel_covid19_withImageNet__malaria_cell_images_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__malaria_cell_images_blog3)
vgg_3channel_covid19_withImageNet__malaria_cell_images_blog3['model'] = 'Vgg19ThreeChannelModelAllLayersCovid19(n_classes=2, learning_rate=1e-3, saved_model="./cellnet_Vgg19_3channel_rotations_covid_withImageNet50000samples_.torch")'
vgg_3channel_covid19_withImageNet__malaria_cell_images_blog3['name'] = 'Vgg19ThreeChannelModelAllLayersCovid19' + "_" + vgg_3channel_covid19_withImageNet__malaria_cell_images_blog3['traindir'] + "_" + "withImageNet"

vgg_3channel_withImageNet_only__malaria_cell_images_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__malaria_cell_images_blog3)
vgg_3channel_withImageNet_only__malaria_cell_images_blog3['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=2, learning_rate=1e-3, pretrained=True)'
vgg_3channel_withImageNet_only__malaria_cell_images_blog3['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg_3channel_withImageNet_only__malaria_cell_images_blog3['traindir'] + "_" + "withImageNet"

vgg_3channel_noImageNet_only__malaria_cell_images_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__malaria_cell_images_blog3)
vgg_3channel_noImageNet_only__malaria_cell_images_blog3['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=2, learning_rate=1e-3, pretrained=False)'
vgg_3channel_noImageNet_only__malaria_cell_images_blog3['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg_3channel_noImageNet_only__malaria_cell_images_blog3['traindir'] + "_" + "withoutImageNet"

cnn_grey_covid19_only__malaria_cell_images_blog3 = copy.deepcopy(vgg_1channel_noImageNet_only__malaria_cell_images_blog3)
cnn_grey_covid19_only__malaria_cell_images_blog3['model'] = 'CNNGreyModelCovid19(n_classes=2, learning_rate=1e-4, saved_model="./cellnet_CNNGrey_1channel_rotations_covid_50000samples_.torch")'
cnn_grey_covid19_only__malaria_cell_images_blog3['name'] = 'CNNGreyModelCovid19' + "_" + cnn_grey_covid19_only__malaria_cell_images_blog3['traindir'] + "_" 

# #########################################################################################################################################################

blood_WBC_baseline_mean = [0.7378, 0.6973, 0.7160]
blood_WBC_baseline_std = [0.0598, 0.1013, 0.0746]
blood_WBC_grey_mean = [0.7116]
blood_WBC_grey_std = [0.0795]

base__blood_WBC_blog3 = copy.deepcopy(base__kaggle_blog3)
base__blood_WBC_blog3['traindir'] = 'blood_WBC'
base__blood_WBC_blog3['testdir'] = 'blood_WBC'
base__blood_WBC_blog3['labels'] = os.listdir(base__blood_WBC_blog3['traindir'])
base__blood_WBC_blog3['train_transforms'] = train_transforms_380_greyscale_blog3
base__blood_WBC_blog3['eval_transforms'] = eval_transforms_380_greyscale_blog3

vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3 = copy.deepcopy(base__blood_WBC_blog3)
vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withoutImageNet50000samples_.torch", n_classes=4, learning_rate=1e-3)'
vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3['traindir'] + "_" + "withoutImageNet"
vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3['train_transforms'][-1] = transforms.Normalize(blood_WBC_grey_mean, blood_WBC_grey_std)
vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3['eval_transforms'][-1] = transforms.Normalize(blood_WBC_grey_mean, blood_WBC_grey_std)

vgg_1channel_covid19_withImageNet__blood_WBC_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3)
vgg_1channel_covid19_withImageNet__blood_WBC_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withImageNet50000samples_.torch", n_classes=4, learning_rate=1e-3)'
vgg_1channel_covid19_withImageNet__blood_WBC_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_withImageNet__blood_WBC_blog3['traindir'] + "_" + "withImageNet"

vgg_1channel_withImageNet_only__blood_WBC_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3)
vgg_1channel_withImageNet_only__blood_WBC_blog3['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=4, learning_rate=1e-3, pretrained=True)'
vgg_1channel_withImageNet_only__blood_WBC_blog3['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_withImageNet_only__blood_WBC_blog3['traindir'] + "_" + "withImageNet"

vgg_1channel_noImageNet_only__blood_WBC_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3)
vgg_1channel_noImageNet_only__blood_WBC_blog3['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=4, learning_rate=1e-3, pretrained=False)'
vgg_1channel_noImageNet_only__blood_WBC_blog3['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_noImageNet_only__blood_WBC_blog3['traindir'] + "_" + "withoutImageNet"

vgg_3channel_covid19_noImageNet__blood_WBC_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3)
vgg_3channel_covid19_noImageNet__blood_WBC_blog3['model'] = 'Vgg19ThreeChannelModelAllLayersCovid19(n_classes=4, learning_rate=1e-3, saved_model="./cellnet_Vgg19_3channel_rotations_covid_withoutImageNet50000samples_.torch")'
vgg_3channel_covid19_noImageNet__blood_WBC_blog3['name'] = 'Vgg19ThreeChannelModelAllLayersCovid19' + "_" + vgg_3channel_covid19_noImageNet__blood_WBC_blog3['traindir'] + "_" + "withoutImageNet"
vgg_3channel_covid19_noImageNet__blood_WBC_blog3['train_transforms'] = train_transforms_224_RGB_blog3
vgg_3channel_covid19_noImageNet__blood_WBC_blog3['eval_transforms'] = eval_transforms_224_RGB_blog3
vgg_3channel_covid19_noImageNet__blood_WBC_blog3['train_transforms'][-1] = transforms.Normalize(blood_WBC_baseline_mean, blood_WBC_baseline_std)
vgg_3channel_covid19_noImageNet__blood_WBC_blog3['eval_transforms'][-1] = transforms.Normalize(blood_WBC_baseline_mean, blood_WBC_baseline_std)

vgg_3channel_covid19_withImageNet__blood_WBC_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__blood_WBC_blog3)
vgg_3channel_covid19_withImageNet__blood_WBC_blog3['model'] = 'Vgg19ThreeChannelModelAllLayersCovid19(n_classes=4, learning_rate=1e-3, saved_model="./cellnet_Vgg19_3channel_rotations_covid_withImageNet50000samples_.torch")'
vgg_3channel_covid19_withImageNet__blood_WBC_blog3['name'] = 'Vgg19ThreeChannelModelAllLayersCovid19' + "_" + vgg_3channel_covid19_withImageNet__blood_WBC_blog3['traindir'] + "_" + "withImageNet"

vgg_3channel_withImageNet_only__blood_WBC_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__blood_WBC_blog3)
vgg_3channel_withImageNet_only__blood_WBC_blog3['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=4, learning_rate=1e-3, pretrained=True)'
vgg_3channel_withImageNet_only__blood_WBC_blog3['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg_3channel_withImageNet_only__blood_WBC_blog3['traindir'] + "_" + "withImageNet"

vgg_3channel_noImageNet_only__blood_WBC_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__blood_WBC_blog3)
vgg_3channel_noImageNet_only__blood_WBC_blog3['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=4, learning_rate=1e-3, pretrained=False)'
vgg_3channel_noImageNet_only__blood_WBC_blog3['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg_3channel_noImageNet_only__blood_WBC_blog3['traindir'] + "_" + "withoutImageNet"

cnn_grey_covid19_only__blood_WBC_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__blood_WBC_images_blog3)
cnn_grey_covid19_only__blood_WBC_blog3['model'] = 'CNNGreyModelCovid19(n_classes=4, learning_rate=1e-4, saved_model="./cellnet_CNNGrey_1channel_rotations_covid_50000samples_.torch")'
cnn_grey_covid19_only__blood_WBC_blog3['name'] = 'CNNGreyModelCovid19' + "_" + cnn_grey_covid19_only__blood_WBC_blog3['traindir'] + "_" 

# #########################################################################################################################################################

cat_counts_baseline_mean = [0.2971, 0.3081, 0.3042]
cat_counts_baseline_std = [0.2778, 0.2874, 0.2818]
cat_counts_grey_mean = [0.3044]
cat_counts_grey_std = [0.2691]


base__cat_counts_blog3 = copy.deepcopy(base__kaggle_blog3)
base__cat_counts_blog3['traindir'] = 'cat_images_border'
base__cat_counts_blog3['testdir'] = 'cat_images_border'
base__cat_counts_blog3['labels'] = os.listdir(base__cat_counts_blog3['traindir'])
base__cat_counts_blog3['train_transforms'] = train_transforms_380_greyscale_blog3
base__cat_counts_blog3['eval_transforms'] = eval_transforms_380_greyscale_blog3

vgg_1channel_covid19_noImageNet__cat_counts_images_blog3 = copy.deepcopy(base__cat_counts_blog3)
vgg_1channel_covid19_noImageNet__cat_counts_images_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withoutImageNet50000samples_.torch", n_classes=5, learning_rate=1e-3)'
vgg_1channel_covid19_noImageNet__cat_counts_images_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_noImageNet__cat_counts_images_blog3['traindir'] + "_" + "withoutImageNet"
vgg_1channel_covid19_noImageNet__cat_counts_images_blog3['train_transforms'][-1] = transforms.Normalize(cat_counts_grey_mean, cat_counts_grey_std)
vgg_1channel_covid19_noImageNet__cat_counts_images_blog3['eval_transforms'][-1] = transforms.Normalize(cat_counts_grey_mean, cat_counts_grey_std)

vgg_1channel_covid19_withImageNet__cat_counts_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__cat_counts_images_blog3)
vgg_1channel_covid19_withImageNet__cat_counts_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withImageNet50000samples_.torch", n_classes=5, learning_rate=1e-3)'
vgg_1channel_covid19_withImageNet__cat_counts_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_withImageNet__cat_counts_blog3['traindir'] + "_" + "withImageNet"

vgg_1channel_withImageNet_only__cat_counts_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__cat_counts_images_blog3)
vgg_1channel_withImageNet_only__cat_counts_blog3['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=5, learning_rate=1e-3, pretrained=True)'
vgg_1channel_withImageNet_only__cat_counts_blog3['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_withImageNet_only__cat_counts_blog3['traindir'] + "_" + "withImageNet"

vgg_1channel_noImageNet_only__cat_counts_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__cat_counts_images_blog3)
vgg_1channel_noImageNet_only__cat_counts_blog3['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=5, learning_rate=1e-3, pretrained=False)'
vgg_1channel_noImageNet_only__cat_counts_blog3['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_noImageNet_only__cat_counts_blog3['traindir'] + "_" + "withoutImageNet"

vgg_3channel_covid19_noImageNet__cat_counts_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__cat_counts_images_blog3)
vgg_3channel_covid19_noImageNet__cat_counts_blog3['model'] = 'Vgg19ThreeChannelModelAllLayersCovid19(n_classes=5, learning_rate=1e-3, saved_model="./cellnet_Vgg19_3channel_rotations_covid_withoutImageNet50000samples_.torch")'
vgg_3channel_covid19_noImageNet__cat_counts_blog3['name'] = 'Vgg19ThreeChannelModelAllLayersCovid19' + "_" + vgg_3channel_covid19_noImageNet__cat_counts_blog3['traindir'] + "_" + "withoutImageNet"
vgg_3channel_covid19_noImageNet__cat_counts_blog3['train_transforms'] = train_transforms_224_RGB_blog3
vgg_3channel_covid19_noImageNet__cat_counts_blog3['eval_transforms'] = eval_transforms_224_RGB_blog3
vgg_3channel_covid19_noImageNet__cat_counts_blog3['train_transforms'][-1] = transforms.Normalize(cat_counts_baseline_mean, cat_counts_baseline_std)
vgg_3channel_covid19_noImageNet__cat_counts_blog3['eval_transforms'][-1] = transforms.Normalize(cat_counts_baseline_mean, cat_counts_baseline_std)

vgg_3channel_covid19_withImageNet__cat_counts_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__cat_counts_blog3)
vgg_3channel_covid19_withImageNet__cat_counts_blog3['model'] = 'Vgg19ThreeChannelModelAllLayersCovid19(n_classes=5, learning_rate=1e-3, saved_model="./cellnet_Vgg19_3channel_rotations_covid_withImageNet50000samples_.torch")'
vgg_3channel_covid19_withImageNet__cat_counts_blog3['name'] = 'Vgg19ThreeChannelModelAllLayersCovid19' + "_" + vgg_3channel_covid19_withImageNet__cat_counts_blog3['traindir'] + "_" + "withImageNet"

vgg_3channel_withImageNet_only__cat_counts_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__cat_counts_blog3)
vgg_3channel_withImageNet_only__cat_counts_blog3['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=5, learning_rate=1e-3, pretrained=True)'
vgg_3channel_withImageNet_only__cat_counts_blog3['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg_3channel_withImageNet_only__cat_counts_blog3['traindir'] + "_" + "withImageNet"

vgg_3channel_noImageNet_only__cat_counts_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__cat_counts_blog3)
vgg_3channel_noImageNet_only__cat_counts_blog3['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=5, learning_rate=1e-3, pretrained=False)'
vgg_3channel_noImageNet_only__cat_counts_blog3['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg_3channel_noImageNet_only__cat_counts_blog3['traindir'] + "_" + "withoutImageNet"

cnn_grey_covid19_only__cat_counts_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__cat_counts_images_blog3)
cnn_grey_covid19_only__cat_counts_blog3['model'] = 'CNNGreyModelCovid19(n_classes=5, learning_rate=1e-4, saved_model="./cellnet_CNNGrey_1channel_rotations_covid_50000samples_.torch")'
cnn_grey_covid19_only__cat_counts_blog3['name'] = 'CNNGreyModelCovid19' + "_" + cnn_grey_covid19_only__cat_counts_blog3['traindir'] + "_" 

# #########################################################################################################################################################

labeled_confocal_protein_baseline_mean = [0.2879, 0.2879, 0.2879]
labeled_confocal_protein_baseline_std = [0.2132, 0.2132, 0.2132]
labeled_confocal_protein_grey_mean = [0.2879]
labeled_confocal_protein_grey_std = [0.2183]

base__labeled_confocal_protein_blog3 = copy.deepcopy(base__kaggle_blog3)
base__labeled_confocal_protein_blog3['traindir'] = 'labeled_confocal_protein'
base__labeled_confocal_protein_blog3['testdir'] = 'labeled_confocal_protein'
base__labeled_confocal_protein_blog3['labels'] = os.listdir(base__labeled_confocal_protein_blog3['traindir'])
base__labeled_confocal_protein_blog3['train_transforms'] = train_transforms_380_greyscale_blog3
base__labeled_confocal_protein_blog3['eval_transforms'] = eval_transforms_380_greyscale_blog3

vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3 = copy.deepcopy(base__labeled_confocal_protein_blog3)
vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withoutImageNet50000samples_.torch", n_classes=9, learning_rate=1e-3)'
vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3['traindir'] + "_" + "withoutImageNet"
vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3['train_transforms'][-1] = transforms.Normalize(labeled_confocal_protein_grey_mean, labeled_confocal_protein_grey_std)
vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3['eval_transforms'][-1] = transforms.Normalize(labeled_confocal_protein_grey_mean, labeled_confocal_protein_grey_std)

vgg_1channel_covid19_withImageNet__labeled_confocal_protein_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3)
vgg_1channel_covid19_withImageNet__labeled_confocal_protein_blog3['model'] = 'Vgg19OneChannelModelAllLayersCovid19(saved_model="./cellnet_Vgg19_1channel_rotations_covid_withImageNet50000samples_.torch", n_classes=9, learning_rate=1e-3)'
vgg_1channel_covid19_withImageNet__labeled_confocal_protein_blog3['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_1channel_covid19_withImageNet__labeled_confocal_protein_blog3['traindir'] + "_" + "withImageNet"

vgg_1channel_withImageNet_only__labeled_confocal_protein_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3)
vgg_1channel_withImageNet_only__labeled_confocal_protein_blog3['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=9, learning_rate=1e-3, pretrained=True)'
vgg_1channel_withImageNet_only__labeled_confocal_protein_blog3['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_withImageNet_only__labeled_confocal_protein_blog3['traindir'] + "_" + "withImageNet"

vgg_1channel_noImageNet_only__labeled_confocal_protein_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3)
vgg_1channel_noImageNet_only__labeled_confocal_protein_blog3['model'] = 'Vgg19OneChannelModelAllLayers(n_classes=9, learning_rate=1e-3, pretrained=False)'
vgg_1channel_noImageNet_only__labeled_confocal_protein_blog3['name'] = 'Vgg19OneChannelModelAllLayers' + "_" + vgg_1channel_noImageNet_only__labeled_confocal_protein_blog3['traindir'] + "_" + "withoutImageNet"

vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3)
vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3['model'] = 'Vgg19ThreeChannelModelAllLayersCovid19(n_classes=9, learning_rate=1e-3, saved_model="./cellnet_Vgg19_3channel_rotations_covid_withoutImageNet50000samples_.torch")'
vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3['name'] = 'Vgg19ThreeChannelModelAllLayersCovid19' + "_" + vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3['traindir'] + "_" + "withoutImageNet"
vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3['train_transforms'] = train_transforms_224_RGB_blog3
vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3['eval_transforms'] = eval_transforms_224_RGB_blog3
vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3['train_transforms'][-1] = transforms.Normalize(labeled_confocal_protein_grey_mean, labeled_confocal_protein_grey_std)
vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3['eval_transforms'][-1] = transforms.Normalize(labeled_confocal_protein_grey_mean, labeled_confocal_protein_grey_std)

vgg_3channel_covid19_withImageNet__labeled_confocal_protein_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3)
vgg_3channel_covid19_withImageNet__labeled_confocal_protein_blog3['model'] = 'Vgg19ThreeChannelModelAllLayersCovid19(n_classes=9, learning_rate=1e-3, saved_model="./cellnet_Vgg19_3channel_rotations_covid_withImageNet50000samples_.torch")'
vgg_3channel_covid19_withImageNet__labeled_confocal_protein_blog3['name'] = 'Vgg19ThreeChannelModelAllLayersCovid19' + "_" + vgg_3channel_covid19_withImageNet__labeled_confocal_protein_blog3['traindir'] + "_" + "withImageNet"

vgg_3channel_withImageNet_only__labeled_confocal_protein_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3)
vgg_3channel_withImageNet_only__labeled_confocal_protein_blog3['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=9, learning_rate=1e-3, pretrained=True)'
vgg_3channel_withImageNet_only__labeled_confocal_protein_blog3['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg_3channel_withImageNet_only__labeled_confocal_protein_blog3['traindir'] + "_" + "withImageNet"

vgg_3channel_noImageNet_only__labeled_confocal_protein_blog3 = copy.deepcopy(vgg_3channel_covid19_noImageNet__labeled_confocal_protein_blog3)
vgg_3channel_noImageNet_only__labeled_confocal_protein_blog3['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=9, learning_rate=1e-3, pretrained=False)'
vgg_3channel_noImageNet_only__labeled_confocal_protein_blog3['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg_3channel_noImageNet_only__labeled_confocal_protein_blog3['traindir'] + "_" + "withoutImageNet"

cnn_grey_covid19_only__labeled_confocal_protein_blog3 = copy.deepcopy(vgg_1channel_covid19_noImageNet__labeled_confocal_protein_images_blog3)
cnn_grey_covid19_only__labeled_confocal_protein_blog3['model'] = 'CNNGreyModelCovid19(n_classes=9, learning_rate=1e-4, saved_model="./cellnet_CNNGrey_1channel_rotations_covid_50000samples_.torch")'
cnn_grey_covid19_only__labeled_confocal_protein_blog3['name'] = 'CNNGreyModelCovid19' + "_" + cnn_grey_covid19_only__labeled_confocal_protein_blog3['traindir'] + "_" 

