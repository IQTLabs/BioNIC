'''

This file is a configuration for all the models tested in our third blog post; image transformations and pre-processing.

Link to blog post: FIXME

'''

import copy
import os
from torchvision import transforms
from augmentations import *

# #########################################################################################################################################################
# BASE TRANSFORMATIONS that will be recycled and modified across several datasets
# #########################################################################################################################################################

resnet_only__malaria_cell_images_baseline_mean = [0.5295, 0.4239, 0.4530]
resnet_only__malaria_cell_images_baseline_std = [0.3366, 0.2723, 0.2876]

train_transforms_malaria_images_blog1 = [
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(size=[224,224]),
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, resnet_only__malaria_cell_images_baseline_std)]

eval_transforms_malaria_images_blog1 = [
        transforms.Resize(size=[224,224]),
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, resnet_only__malaria_cell_images_baseline_std)
        ]

# #########################################################################################################################################################
train_transforms_malaria_images_fullsize = [
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, resnet_only__malaria_cell_images_baseline_std)]

eval_transforms_malaria_images_fullsize = [
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, resnet_only__malaria_cell_images_baseline_std)
        ]

train_transforms_malaria_images_halfsize = [
        transforms.Resize(size=[500,500]),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, resnet_only__malaria_cell_images_baseline_std)]

eval_transforms_malaria_images_halfsize = [
        transforms.Resize(size=[500,500]),
        transforms.ToTensor(),
        transforms.Normalize(resnet_only__malaria_cell_images_baseline_mean, resnet_only__malaria_cell_images_baseline_std)
        ]

# #########################################################################################################################################################
# MODEL SPECIFICATIONS for models in blog posts 1 through 3
# #########################################################################################################################################################

vgg_only__kaggle_baseline_mean = [0.0918, 0.0918, 0.0918]
vgg_only__kaggle_baseline_std = [0.1675, 0.1675, 0.1675]

vgg_only__kaggle_blog1 = {}
vgg_only__kaggle_blog1['traindir'] = 'kaggle_bowl'
vgg_only__kaggle_blog1['testdir'] = 'kaggle_bowl'
vgg_only__kaggle_blog1['labels'] = os.listdir(vgg_only__kaggle_blog1['traindir'])
vgg_only__kaggle_blog1['model'] = 'VggModelAllLayers(freeze=False,  n_classes=8, learning_rate=learning_rate)'
vgg_only__kaggle_blog1['max_class_samples'] = None #400
vgg_only__kaggle_blog1['train_transforms'] = train_transforms_malaria_images_blog1
vgg_only__kaggle_blog1['eval_transforms'] = eval_transforms_malaria_images_blog1
vgg_only__kaggle_blog1['train_transforms'][-1] = transforms.Normalize(vgg_only__kaggle_baseline_mean, vgg_only__kaggle_baseline_std)
vgg_only__kaggle_blog1['eval_transforms'][-1] = transforms.Normalize(vgg_only__kaggle_baseline_mean, vgg_only__kaggle_baseline_std)
vgg_only__kaggle_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_only__kaggle_blog1['traindir'] + "_" + "blog1"
vgg_only__kaggle_blog1['usesCV'] = True

vgg_covid19_only__kaggle_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
vgg_covid19_only__kaggle_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(n_classes=8, learning_rate=learning_rate)'
vgg_covid19_only__kaggle_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_covid19_only__kaggle_blog1['traindir'] + "_" + "blog1"
vgg_covid19_only__kaggle_blog1['train_transforms'].insert(len(vgg_covid19_only__kaggle_blog1['train_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__kaggle_blog1['eval_transforms'].insert(len(vgg_covid19_only__kaggle_blog1['eval_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__kaggle_blog1['train_transforms'][-1] = transforms.Normalize([0.0918], [0.1675])
vgg_covid19_only__kaggle_blog1['eval_transforms'][-1] = transforms.Normalize([0.0918], [0.1675])

vgg19_pretrained_only__kaggle_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
vgg19_pretrained_only__kaggle_blog1['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=8, learning_rate=learning_rate)'
vgg19_pretrained_only__kaggle_blog1['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg19_pretrained_only__kaggle_blog1['traindir'] + "_" + "blog1"

vgg16BN_pretrained_only__kaggle_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
vgg16BN_pretrained_only__kaggle_blog1['model'] = 'Vgg16BNThreeChannelModelAllLayers(n_classes=8, learning_rate=learning_rate)'
vgg16BN_pretrained_only__kaggle_blog1['name'] = 'Vgg16BNThreeChannelModelAllLayers' + "_" + vgg16BN_pretrained_only__kaggle_blog1['traindir'] + "_" + "blog1"

vgg_lower__kaggle_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
vgg_lower__kaggle_blog1['model'] = 'VggModelLowerLayers(freeze=False, n_classes=8, learning_rate=learning_rate)'
vgg_lower__kaggle_blog1['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__kaggle_blog1['traindir'] + "_" + "blog1"

vgg_untrained__kaggle_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
vgg_untrained__kaggle_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=8, learning_rate=learning_rate, pretrained=False)'
vgg_untrained__kaggle_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_untrained__kaggle_blog1['traindir'] + "_" + "blog1_untrained"

resnet_only__kaggle_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_only__kaggle_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=8, learning_rate=learning_rate)'
resnet_only__kaggle_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__kaggle_blog1['traindir'] + "_" + "blog1"

resnet_untrained__kaggle_blog1 = copy.deepcopy(resnet_only__kaggle_blog1)
resnet_untrained__kaggle_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=8, learning_rate=learning_rate, pretrained=False)'
resnet_untrained__kaggle_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_untrained__kaggle_blog1['traindir'] + "_" + "blog1_untrained"

resnet_lower__kaggle_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_lower__kaggle_blog1['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=8, learning_rate=learning_rate)'
resnet_lower__kaggle_blog1['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__kaggle_blog1['traindir'] + "_" + "blog1"

cnn_only__kaggle_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
cnn_only__kaggle_blog1['model'] = 'CNNModel(n_classes=8, learning_rate=learning_rate)'
cnn_only__kaggle_blog1['name'] = 'CNNModel' + "_" + cnn_only__kaggle_blog1['traindir'] + "_" + "blog1"

cnn_mini__kaggle_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
cnn_mini__kaggle_blog1['model'] = 'CNNMiniModel(n_classes=8, learning_rate=learning_rate)'
cnn_mini__kaggle_blog1['name'] = 'CNNMiniModel' + "_" + cnn_mini__kaggle_blog1['traindir'] + "_" + "blog1"

cnn_only__kaggle_fullsize = copy.deepcopy(cnn_only__kaggle_blog1)
cnn_only__kaggle_fullsize['train_transforms'] = train_transforms_malaria_images_fullsize
cnn_only__kaggle_fullsize['eval_transforms'] = eval_transforms_malaria_images_fullsize
cnn_only__kaggle_fullsize['name'] = 'CNNModel' + "_" + cnn_only__kaggle_fullsize['traindir'] + "_" + "fullsize"

resnet_only__kaggle_tophat_otsu_thresh = copy.deepcopy(resnet_only__kaggle_blog1)
resnet_only__kaggle_tophat_otsu_thresh['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__kaggle_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
resnet_only__kaggle_tophat_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False))
resnet_only__kaggle_tophat_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False))
resnet_only__kaggle_tophat_otsu_thresh['train_transforms'].insert(0, TopHat())
resnet_only__kaggle_tophat_otsu_thresh['eval_transforms'].insert(0, TopHat())

resnet_only__kaggle_tophat_mean_thresh = copy.deepcopy(resnet_only__kaggle_blog1)
resnet_only__kaggle_tophat_mean_thresh['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__kaggle_tophat_mean_thresh['traindir'] + "_" + "tophat_mean_thresh"
resnet_only__kaggle_tophat_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True))
resnet_only__kaggle_tophat_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True))
resnet_only__kaggle_tophat_mean_thresh['train_transforms'].insert(0, TopHat())
resnet_only__kaggle_tophat_mean_thresh['eval_transforms'].insert(0, TopHat())

resnet_only__kaggle_sobel_otsu_thresh = copy.deepcopy(resnet_only__kaggle_blog1)
resnet_only__kaggle_sobel_otsu_thresh['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__kaggle_sobel_otsu_thresh['traindir'] + "_" + "otsu_thresh_sobel"
resnet_only__kaggle_sobel_otsu_thresh['train_transforms'].insert(0, SobelX())
resnet_only__kaggle_sobel_otsu_thresh['eval_transforms'].insert(0, SobelX())
resnet_only__kaggle_sobel_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False))
resnet_only__kaggle_sobel_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False))

resnet_only__kaggle_sobel_mean_thresh = copy.deepcopy(resnet_only__kaggle_blog1)
resnet_only__kaggle_sobel_mean_thresh['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__kaggle_sobel_mean_thresh['traindir'] + "_" + "mean_thresh_sobel"
resnet_only__kaggle_sobel_mean_thresh['train_transforms'].insert(0, SobelX())
resnet_only__kaggle_sobel_mean_thresh['eval_transforms'].insert(0, SobelX())
resnet_only__kaggle_sobel_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True))
resnet_only__kaggle_sobel_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True))

resnet_only__kaggle_erosion = copy.deepcopy(resnet_only__kaggle_blog1)
resnet_only__kaggle_erosion['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__kaggle_erosion['traindir'] + "_" + "erosion"
resnet_only__kaggle_erosion['train_transforms'].insert(0, ErosionAndDilation())
resnet_only__kaggle_erosion['eval_transforms'].insert(0, ErosionAndDilation())

resnet_only__kaggle_clahe = copy.deepcopy(resnet_only__kaggle_blog1)
resnet_only__kaggle_clahe['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__kaggle_clahe['traindir'] + "_" + "clahe"
resnet_only__kaggle_clahe['train_transforms'].insert(0, ContrastThroughHistogram())
resnet_only__kaggle_clahe['eval_transforms'].insert(0, ContrastThroughHistogram())

vgg_lower__kaggle_LBP = copy.deepcopy(vgg_lower__kaggle_blog1)
vgg_lower__kaggle_LBP['name'] = 'ResNet18ModelAllLayers' + "_" + vgg_lower__kaggle_LBP['traindir'] + "_" + "LBP"
vgg_lower__kaggle_LBP['train_transforms'].insert(0, LocalBinaryPattern())
vgg_lower__kaggle_LBP['eval_transforms'].insert(0, LocalBinaryPattern())

# #########################################################################################################################################################
'''
biomed_self_label_shape_baseline_mean = [0.4073, 0.4073, 0.4073]
biomed_self_label_shape_baseline_std = [0.1076, 0.1076, 0.1076]

biomed_self_label_shape_blog1_resnetfull = copy.deepcopy(vgg_only__kaggle_blog1)
biomed_self_label_shape_blog1_resnetfull['traindir'] = 'biomed_self_label_shape'
biomed_self_label_shape_blog1_resnetfull['testdir'] = 'biomed_self_label_shape'
biomed_self_label_shape_blog1_resnetfull['labels'] = os.listdir(biomed_self_label_shape_blog1_resnetfull['traindir'])
biomed_self_label_shape_blog1_resnetfull['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
biomed_self_label_shape_blog1_resnetfull['name'] = 'ResNet18ModelAllLayers' + "_" + biomed_self_label_shape_blog1_resnetfull['traindir'] + "_" + "blog1_resnetfull"
biomed_self_label_shape_blog1_resnetfull['train_transforms'][-1] = transforms.Normalize(biomed_self_label_shape_baseline_mean, biomed_self_label_shape_baseline_std)
biomed_self_label_shape_blog1_resnetfull['eval_transforms'][-1] = transforms.Normalize(biomed_self_label_shape_baseline_mean, biomed_self_label_shape_baseline_std)
biomed_self_label_shape_blog1_resnetfull['max_class_samples'] = 500

biomed_self_label_shape_blog1_resnetlower = copy.deepcopy(biomed_self_label_shape_blog1_resnetfull)
biomed_self_label_shape_blog1_resnetlower['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
biomed_self_label_shape_blog1_resnetlower['name'] = 'ResNet18ModelLowerLayers' + "_" + biomed_self_label_shape_blog1_resnetlower['traindir'] + "_" + "blog1_resnetlower"

biomed_self_label_shape_blog1_vggfull = copy.deepcopy(biomed_self_label_shape_blog1_resnetfull)
biomed_self_label_shape_blog1_vggfull['model'] = 'VggModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
biomed_self_label_shape_blog1_vggfull['name'] = 'VggModelAllLayers' + "_" + biomed_self_label_shape_blog1_vggfull['traindir'] + "_" + "blog1_vggfull"

biomed_self_label_shape_blog1_vgglower = copy.deepcopy(biomed_self_label_shape_blog1_resnetfull)
biomed_self_label_shape_blog1_vgglower['model'] = 'VggModelLowerLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
biomed_self_label_shape_blog1_vgglower['name'] = 'VggModelLowerLayers' + "_" + biomed_self_label_shape_blog1_vgglower['traindir'] + "_" + "blog1_vgglower"

biomed_self_label_shape_blog1_cnnfull = copy.deepcopy(biomed_self_label_shape_blog1_resnetfull)
biomed_self_label_shape_blog1_cnnfull['model'] = 'CNNModel(n_classes=4, learning_rate=learning_rate)'
biomed_self_label_shape_blog1_cnnfull['name'] = 'CNNModel' + "_" + biomed_self_label_shape_blog1_cnnfull['traindir'] + "_" + "blog1_cnnfull"

biomed_self_label_shape_blog1_cnnmini = copy.deepcopy(biomed_self_label_shape_blog1_resnetfull)
biomed_self_label_shape_blog1_cnnmini['model'] = 'CNNMiniModel(n_classes=4, learning_rate=learning_rate)'
biomed_self_label_shape_blog1_cnnmini['name'] = 'CNNMiniModel' + "_" + biomed_self_label_shape_blog1_cnnmini['traindir'] + "_" + "blog1_cnnmini"

biomed_self_label_shape_blog1_cnngrey = copy.deepcopy(biomed_self_label_shape_blog1_resnetfull)
biomed_self_label_shape_blog1_cnngrey['model'] = 'CNNGreyModel(n_classes=4, learning_rate=learning_rate)'
biomed_self_label_shape_blog1_cnngrey['name'] = 'CNNGreyModel' + "_" + biomed_self_label_shape_blog1_cnngrey['traindir'] + "_" + "blog1_cnngrey"
biomed_self_label_shape_blog1_cnngrey['train_transforms'] = train_transforms_kaggle_bowl_blog1_grey
biomed_self_label_shape_blog1_cnngrey['eval_transforms'] = eval_transforms_kaggle_bowl_center_crop_grey

vgg_only__biomed_self_label_shape_centercrop = copy.deepcopy(vgg_only__kaggle_center_crop)
vgg_only__biomed_self_label_shape_centercrop['traindir'] = 'biomed_self_label_shape'
vgg_only__biomed_self_label_shape_centercrop['testdir'] = 'biomed_self_label_shape'
vgg_only__biomed_self_label_shape_centercrop['labels'] = os.listdir(vgg_only__biomed_self_label_shape_centercrop['traindir'])
vgg_only__biomed_self_label_shape_centercrop['model'] = 'VggModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
vgg_only__biomed_self_label_shape_centercrop['name'] = 'VggModelAllLayers' + "_" + vgg_only__biomed_self_label_shape_centercrop['traindir'] + "_" + "centercrop"
vgg_only__biomed_self_label_shape_centercrop['train_transforms'][-1] = transforms.Normalize(biomed_self_label_shape_baseline_mean, biomed_self_label_shape_baseline_std)
vgg_only__biomed_self_label_shape_centercrop['eval_transforms'][-1] = transforms.Normalize(biomed_self_label_shape_baseline_mean, biomed_self_label_shape_baseline_std)
vgg_only__biomed_self_label_shape_centercrop['max_class_samples'] = 500

vgg_only__biomed_self_label_shape_randomcrop = copy.deepcopy(vgg_only__kaggle_random_crop)
vgg_only__biomed_self_label_shape_randomcrop['traindir'] = 'biomed_self_label_shape'
vgg_only__biomed_self_label_shape_randomcrop['testdir'] = 'biomed_self_label_shape'
vgg_only__biomed_self_label_shape_randomcrop['labels'] = os.listdir(vgg_only__biomed_self_label_shape_randomcrop['traindir'])
vgg_only__biomed_self_label_shape_randomcrop['model'] = 'VggModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
vgg_only__biomed_self_label_shape_randomcrop['name'] = 'VggModelAllLayers' + "_" + vgg_only__biomed_self_label_shape_randomcrop['traindir'] + "_" + "randomcrop"
vgg_only__biomed_self_label_shape_randomcrop['train_transforms'][-1] = transforms.Normalize(biomed_self_label_shape_baseline_mean, biomed_self_label_shape_baseline_std)
vgg_only__biomed_self_label_shape_randomcrop['eval_transforms'][-1] = transforms.Normalize(biomed_self_label_shape_baseline_mean, biomed_self_label_shape_baseline_std)
vgg_only__biomed_self_label_shape_randomcrop['max_class_samples'] = 500

vgg_only__biomed_self_label_shape_full_resize = copy.deepcopy(vgg_only__kaggle_center_full_resize)
vgg_only__biomed_self_label_shape_full_resize['traindir'] = 'biomed_self_label_shape'
vgg_only__biomed_self_label_shape_full_resize['testdir'] = 'biomed_self_label_shape'
vgg_only__biomed_self_label_shape_full_resize['labels'] = os.listdir(vgg_only__biomed_self_label_shape_full_resize['traindir'])
vgg_only__biomed_self_label_shape_full_resize['model'] = 'VggModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
vgg_only__biomed_self_label_shape_full_resize['name'] = 'VggModelAllLayers' + "_" + vgg_only__biomed_self_label_shape_full_resize['traindir'] + "_" + "full_resize"
vgg_only__biomed_self_label_shape_full_resize['train_transforms'][-1] = transforms.Normalize(biomed_self_label_shape_baseline_mean, biomed_self_label_shape_baseline_std)
vgg_only__biomed_self_label_shape_full_resize['eval_transforms'][-1] = transforms.Normalize(biomed_self_label_shape_baseline_mean, biomed_self_label_shape_baseline_std)
vgg_only__biomed_self_label_shape_full_resize['max_class_samples'] = 500

vgg_only__biomed_self_label_shape_half_resize = copy.deepcopy(vgg_only__kaggle_center_half_resize)
vgg_only__biomed_self_label_shape_half_resize['traindir'] = 'biomed_self_label_shape'
vgg_only__biomed_self_label_shape_half_resize['testdir'] = 'biomed_self_label_shape'
vgg_only__biomed_self_label_shape_half_resize['labels'] = os.listdir(vgg_only__biomed_self_label_shape_half_resize['traindir'])
vgg_only__biomed_self_label_shape_half_resize['model'] = 'VggModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
vgg_only__biomed_self_label_shape_half_resize['name'] = 'VggModelAllLayers' + "_" + vgg_only__biomed_self_label_shape_half_resize['traindir'] + "_" + "half_resize"
vgg_only__biomed_self_label_shape_half_resize['train_transforms'][-1] = transforms.Normalize(biomed_self_label_shape_baseline_mean, biomed_self_label_shape_baseline_std)
vgg_only__biomed_self_label_shape_half_resize['eval_transforms'][-1] = transforms.Normalize(biomed_self_label_shape_baseline_mean, biomed_self_label_shape_baseline_std)
vgg_only__biomed_self_label_shape_half_resize['max_class_samples'] = 500

vgg_only__biomed_self_label_shape_tophat_otsu_thresh = copy.deepcopy(vgg_only__biomed_self_label_shape_full_resize)
vgg_only__biomed_self_label_shape_tophat_otsu_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__biomed_self_label_shape_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
vgg_only__biomed_self_label_shape_tophat_otsu_thresh['train_transforms'].insert(1, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))
vgg_only__biomed_self_label_shape_tophat_otsu_thresh['eval_transforms'].insert(1, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))
vgg_only__biomed_self_label_shape_tophat_otsu_thresh['train_transforms'].insert(1, TopHat())
vgg_only__biomed_self_label_shape_tophat_otsu_thresh['eval_transforms'].insert(1, TopHat())

vgg_only__biomed_self_label_shape_tophat_mean_adaptive_thresh = copy.deepcopy(vgg_only__biomed_self_label_shape_full_resize)
vgg_only__biomed_self_label_shape_tophat_mean_adaptive_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__biomed_self_label_shape_tophat_mean_adaptive_thresh['traindir'] + "_" + "tophat_mean_adaptive_thresh"
vgg_only__biomed_self_label_shape_tophat_mean_adaptive_thresh['train_transforms'].insert(1, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))
vgg_only__biomed_self_label_shape_tophat_mean_adaptive_thresh['eval_transforms'].insert(1, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))
vgg_only__biomed_self_label_shape_tophat_mean_adaptive_thresh['train_transforms'].insert(1, TopHat())
vgg_only__biomed_self_label_shape_tophat_mean_adaptive_thresh['eval_transforms'].insert(1, TopHat())

vgg_only__biomed_self_label_shape_sobel_otsu_thresh = copy.deepcopy(vgg_only__biomed_self_label_shape_full_resize)
vgg_only__biomed_self_label_shape_sobel_otsu_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__biomed_self_label_shape_sobel_otsu_thresh['traindir'] + "_" + "sobel_otsu_thresh"
vgg_only__biomed_self_label_shape_sobel_otsu_thresh['train_transforms'].insert(1, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))
vgg_only__biomed_self_label_shape_sobel_otsu_thresh['eval_transforms'].insert(1, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))
vgg_only__biomed_self_label_shape_sobel_otsu_thresh['train_transforms'].insert(1, SobelX())
vgg_only__biomed_self_label_shape_sobel_otsu_thresh['eval_transforms'].insert(1, SobelX())

vgg_only__biomed_self_label_shape_sobel_mean_adaptive_thresh = copy.deepcopy(vgg_only__biomed_self_label_shape_full_resize)
vgg_only__biomed_self_label_shape_sobel_mean_adaptive_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__biomed_self_label_shape_sobel_mean_adaptive_thresh['traindir'] + "_" + "sobel_mean_adaptive_thresh"
vgg_only__biomed_self_label_shape_sobel_mean_adaptive_thresh['train_transforms'].insert(1, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))
vgg_only__biomed_self_label_shape_sobel_mean_adaptive_thresh['eval_transforms'].insert(1, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))
vgg_only__biomed_self_label_shape_sobel_mean_adaptive_thresh['train_transforms'].insert(1, SobelX())
vgg_only__biomed_self_label_shape_sobel_mean_adaptive_thresh['eval_transforms'].insert(1, SobelX())

vgg_only__biomed_self_label_shape_erode = copy.deepcopy(vgg_only__biomed_self_label_shape_full_resize)
vgg_only__biomed_self_label_shape_erode['name'] = 'VggModelAllLayers' + "_" + vgg_only__biomed_self_label_shape_erode['traindir'] + "_" + "erode"
vgg_only__biomed_self_label_shape_erode['train_transforms'].insert(1, ErosionAndDilation())
vgg_only__biomed_self_label_shape_erode['eval_transforms'].insert(1, ErosionAndDilation())

vgg_only__biomed_self_label_shape_LBP = copy.deepcopy(vgg_only__biomed_self_label_shape_full_resize)
vgg_only__biomed_self_label_shape_LBP['name'] = 'VggModelAllLayers' + "_" + vgg_only__biomed_self_label_shape_LBP['traindir'] + "_" + "LBP"
vgg_only__biomed_self_label_shape_LBP['train_transforms'].insert(1, LocalBinaryPattern())
vgg_only__biomed_self_label_shape_LBP['eval_transforms'].insert(1, LocalBinaryPattern())'''

# #########################################################################################################################################################

resnet_only__malaria_cell_images_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_only__malaria_cell_images_blog1['traindir'] = 'malaria_cell_images'
resnet_only__malaria_cell_images_blog1['testdir'] = 'malaria_cell_images'
resnet_only__malaria_cell_images_blog1['labels'] = os.listdir(resnet_only__malaria_cell_images_blog1['traindir'])
resnet_only__malaria_cell_images_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=2, learning_rate=learning_rate)'
resnet_only__malaria_cell_images_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__malaria_cell_images_blog1['traindir'] + "_" + "blog1"
resnet_only__malaria_cell_images_blog1['train_transforms'] = train_transforms_malaria_images_blog1
resnet_only__malaria_cell_images_blog1['eval_transforms'] = eval_transforms_malaria_images_blog1
resnet_only__malaria_cell_images_blog1['max_class_samples'] = 500

vgg_covid19_only__malaria_cell_images_blog1 = copy.deepcopy(resnet_only__malaria_cell_images_blog1)
vgg_covid19_only__malaria_cell_images_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(n_classes=2, learning_rate=learning_rate)'
vgg_covid19_only__malaria_cell_images_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_covid19_only__malaria_cell_images_blog1['traindir'] + "_" + "blog1"
vgg_covid19_only__malaria_cell_images_blog1['train_transforms'].insert(len(vgg_covid19_only__malaria_cell_images_blog1['train_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__malaria_cell_images_blog1['eval_transforms'].insert(len(vgg_covid19_only__malaria_cell_images_blog1['eval_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__malaria_cell_images_blog1['train_transforms'][-1] = transforms.Normalize([0.4588], [0.2844])
vgg_covid19_only__malaria_cell_images_blog1['eval_transforms'][-1] = transforms.Normalize([0.4588], [0.2844])

vgg19_pretrained_only__malaria_cell_images_blog1 = copy.deepcopy(resnet_only__malaria_cell_images_blog1)
vgg19_pretrained_only__malaria_cell_images_blog1['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=2, learning_rate=learning_rate)'
vgg19_pretrained_only__malaria_cell_images_blog1['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg19_pretrained_only__malaria_cell_images_blog1['traindir'] + "_" + "blog1"

vgg16BN_pretrained_only__malaria_cell_images_blog1 = copy.deepcopy(vgg19_pretrained_only__malaria_cell_images_blog1)
vgg16BN_pretrained_only__malaria_cell_images_blog1['model'] = 'Vgg16BNThreeChannelModelAllLayers(n_classes=2, learning_rate=learning_rate)'
vgg16BN_pretrained_only__malaria_cell_images_blog1['name'] = 'Vgg16BNThreeChannelModelAllLayers' + "_" + vgg16BN_pretrained_only__malaria_cell_images_blog1['traindir'] + "_" + "blog1"

resnet_untrained__malaria_cell_images_blog1 = copy.deepcopy(resnet_only__malaria_cell_images_blog1)
resnet_untrained__malaria_cell_images_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=2, learning_rate=learning_rate, pretrained=False)'
resnet_untrained__malaria_cell_images_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_untrained__malaria_cell_images_blog1['traindir'] + "_" + "blog1_untrained"

resnet_lower__malaria_cell_images_blog1 = copy.deepcopy(resnet_only__malaria_cell_images_blog1)
resnet_lower__malaria_cell_images_blog1['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=2, learning_rate=learning_rate)'
resnet_lower__malaria_cell_images_blog1['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_only__malaria_cell_images_blog1['traindir'] + "_" + "blog1"

vgg_only__malaria_cell_images_blog1 = copy.deepcopy(resnet_only__malaria_cell_images_blog1)
vgg_only__malaria_cell_images_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=2, learning_rate=learning_rate)'
vgg_only__malaria_cell_images_blog1['name'] = 'VggModelAllLayers' + "_" + resnet_only__malaria_cell_images_blog1['traindir'] + "_" + "blog1"

vgg_untrained__malaria_cell_images_blog1 = copy.deepcopy(vgg_only__malaria_cell_images_blog1)
vgg_untrained__malaria_cell_images_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=2, learning_rate=learning_rate, pretrained=False)'
vgg_untrained__malaria_cell_images_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_untrained__malaria_cell_images_blog1['traindir'] + "_" + "blog1_untrained"

vgg_lower__malaria_cell_images_blog1 = copy.deepcopy(resnet_only__malaria_cell_images_blog1)
vgg_lower__malaria_cell_images_blog1['model'] = 'VggModelLowerLayers(freeze=False, n_classes=2, learning_rate=learning_rate)'
vgg_lower__malaria_cell_images_blog1['name'] = 'VggModelLowerLayers' + "_" + resnet_only__malaria_cell_images_blog1['traindir'] + "_" + "blog1"

cnn_only__malaria_cell_images_blog1 = copy.deepcopy(resnet_only__malaria_cell_images_blog1)
cnn_only__malaria_cell_images_blog1['model'] = 'CNNModel(n_classes=2, learning_rate=learning_rate)'
cnn_only__malaria_cell_images_blog1['name'] = 'CNNModel' + "_" + resnet_only__malaria_cell_images_blog1['traindir'] + "_" + "blog1"

cnn_mini__malaria_cell_images_blog1 = copy.deepcopy(resnet_only__malaria_cell_images_blog1)
cnn_mini__malaria_cell_images_blog1['model'] = 'CNNMiniModel(n_classes=2, learning_rate=learning_rate)'
cnn_mini__malaria_cell_images_blog1['name'] = 'CNNMiniModel' + "_" + resnet_only__malaria_cell_images_blog1['traindir'] + "_" + "blog1"

cnn_only__malaria_cell_fullsize = copy.deepcopy(cnn_only__malaria_cell_images_blog1)
cnn_only__malaria_cell_fullsize['train_transforms'] = train_transforms_malaria_images_fullsize
cnn_only__malaria_cell_fullsize['eval_transforms'] = eval_transforms_malaria_images_fullsize
cnn_only__malaria_cell_fullsize['name'] = 'CNNModel' + "_" + cnn_only__malaria_cell_fullsize['traindir'] + "_" + "fullsize"

'''cnn_only__malaria_cell_images_full_resize = copy.deepcopy(vgg_only__kaggle_center_full_resize)
cnn_only__malaria_cell_images_full_resize['traindir'] = 'malaria_cell_images'
cnn_only__malaria_cell_images_full_resize['testdir'] = 'malaria_cell_images'
cnn_only__malaria_cell_images_full_resize['labels'] = os.listdir(cnn_only__malaria_cell_images_blog1['traindir'])
cnn_only__malaria_cell_images_full_resize['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=2, learning_rate=learning_rate)'
cnn_only__malaria_cell_images_full_resize['name'] = 'ResNet18ModelAllLayers' + "_" + cnn_only__malaria_cell_images_full_resize['traindir'] + "_" + "full_resize"
cnn_only__malaria_cell_images_full_resize['train_transforms'][-1] = transforms.Normalize(cnn_only__malaria_cell_images_baseline_mean, cnn_only__malaria_cell_images_baseline_std)
cnn_only__malaria_cell_images_full_resize['eval_transforms'][-1] = transforms.Normalize(cnn_only__malaria_cell_images_baseline_mean, cnn_only__malaria_cell_images_baseline_std)
cnn_only__malaria_cell_images_full_resize['epochs'] = 3
cnn_only__malaria_cell_images_full_resize['max_class_samples'] = 500'''

cnn_only__malaria_cell_images_tophat_otsu_thresh = copy.deepcopy(cnn_only__malaria_cell_images_blog1)
cnn_only__malaria_cell_images_tophat_otsu_thresh['name'] = 'CNNModel' + "_" + cnn_only__malaria_cell_images_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
cnn_only__malaria_cell_images_tophat_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=3))
cnn_only__malaria_cell_images_tophat_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=3))
cnn_only__malaria_cell_images_tophat_otsu_thresh['train_transforms'].insert(0, TopHat())
cnn_only__malaria_cell_images_tophat_otsu_thresh['eval_transforms'].insert(0, TopHat())

cnn_only__malaria_cell_images_sobel_otsu_thresh = copy.deepcopy(cnn_only__malaria_cell_images_blog1)
cnn_only__malaria_cell_images_sobel_otsu_thresh['name'] = 'CNNModel' + "_" + cnn_only__malaria_cell_images_sobel_otsu_thresh['traindir'] + "_" + "otsu_thresh_sobel"
cnn_only__malaria_cell_images_sobel_otsu_thresh['train_transforms'].insert(0, SobelX())
cnn_only__malaria_cell_images_sobel_otsu_thresh['eval_transforms'].insert(0, SobelX())
cnn_only__malaria_cell_images_sobel_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=3))
cnn_only__malaria_cell_images_sobel_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=3))

cnn_only__malaria_cell_images_sobel_mean_thresh = copy.deepcopy(cnn_only__malaria_cell_images_blog1)
cnn_only__malaria_cell_images_sobel_mean_thresh['name'] = 'CNNModel' + "_" + cnn_only__malaria_cell_images_sobel_mean_thresh['traindir'] + "_" + "mean_thresh_sobel"
cnn_only__malaria_cell_images_sobel_mean_thresh['train_transforms'].insert(0, SobelX())
cnn_only__malaria_cell_images_sobel_mean_thresh['eval_transforms'].insert(0, SobelX())
cnn_only__malaria_cell_images_sobel_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=3))
cnn_only__malaria_cell_images_sobel_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=3))

cnn_only__malaria_cell_images_tophat_mean_thresh = copy.deepcopy(cnn_only__malaria_cell_images_blog1)
cnn_only__malaria_cell_images_tophat_mean_thresh['name'] = 'CNNModel' + "_" + cnn_only__malaria_cell_images_tophat_mean_thresh['traindir'] + "_" + "tophat_mean_thresh"
cnn_only__malaria_cell_images_tophat_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=3))
cnn_only__malaria_cell_images_tophat_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=3))
cnn_only__malaria_cell_images_tophat_mean_thresh['train_transforms'].insert(0, TopHat())
cnn_only__malaria_cell_images_tophat_mean_thresh['eval_transforms'].insert(0, TopHat())

cnn_only__malaria_cell_images_erode = copy.deepcopy(cnn_only__malaria_cell_images_blog1)
cnn_only__malaria_cell_images_erode['name'] = 'CNNModel' + "_" + cnn_only__malaria_cell_images_erode['traindir'] + "_" + "erode"
cnn_only__malaria_cell_images_erode['train_transforms'].insert(0, ErosionAndDilation())
cnn_only__malaria_cell_images_erode['eval_transforms'].insert(0, ErosionAndDilation())

cnn_only__malaria_cell_images_clahe = copy.deepcopy(cnn_only__malaria_cell_images_blog1)
cnn_only__malaria_cell_images_clahe['name'] = 'CNNModel' + "_" + cnn_only__malaria_cell_images_clahe['traindir'] + "_" + "clahe"
cnn_only__malaria_cell_images_clahe['train_transforms'].insert(0, ContrastThroughHistogram())
cnn_only__malaria_cell_images_clahe['eval_transforms'].insert(0, ContrastThroughHistogram())

cnn_only__malaria_cell_images_LBP = copy.deepcopy(cnn_only__malaria_cell_images_blog1)
cnn_only__malaria_cell_images_LBP['name'] = 'CNNModel' + "_" + cnn_only__malaria_cell_images_LBP['traindir'] + "_" + "LBP"
cnn_only__malaria_cell_images_LBP['train_transforms'].insert(0, LocalBinaryPattern())
cnn_only__malaria_cell_images_LBP['eval_transforms'].insert(0, LocalBinaryPattern())

resnet_only__malaria_cell_images_blog1_20e = copy.deepcopy(resnet_only__malaria_cell_images_blog1)
resnet_only__malaria_cell_images_blog1_20e['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__malaria_cell_images_blog1_20e['traindir'] + "_" + "blog1_20e"
resnet_only__malaria_cell_images_blog1_20e['epochs'] = 20

cnn_only__malaria_cell_images_venn_blog1 = copy.deepcopy(cnn_only__malaria_cell_images_blog1)
cnn_only__malaria_cell_images_venn_blog1['traindir'] = 'malaria_cell_images_subset_500'
cnn_only__malaria_cell_images_venn_blog1['testdir'] = 'malaria_cell_images_subset_500'
cnn_only__malaria_cell_images_venn_blog1['name'] = 'CNNModel' + "_" + cnn_only__malaria_cell_images_venn_blog1['traindir'] + "_" + "blog1_venn"
cnn_only__malaria_cell_images_venn_blog1['usesCV'] = False
cnn_only__malaria_cell_images_venn_blog1['holdout'] = 'HOLDOUT_malaria_cell_images'
cnn_only__malaria_cell_images_venn_blog1['venn_data'] = './CNNModel_malaria_cell_images_subset_500_blog1_venn_venn_data.results364.csv'

# #########################################################################################################################################################
blood_WBC_baseline_mean = [0.7378, 0.6973, 0.7160]
blood_WBC_baseline_std = [0.0598, 0.1013, 0.0746]

resnet_only__WBC_images_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_only__WBC_images_blog1['traindir'] = 'blood_WBC'
resnet_only__WBC_images_blog1['testdir'] = 'blood_WBC'
resnet_only__WBC_images_blog1['labels'] = os.listdir(resnet_only__WBC_images_blog1['traindir'])
resnet_only__WBC_images_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
resnet_only__WBC_images_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__WBC_images_blog1['traindir'] + "_" + "blog1"
resnet_only__WBC_images_blog1['train_transforms'] = train_transforms_malaria_images_blog1
resnet_only__WBC_images_blog1['eval_transforms'] = eval_transforms_malaria_images_blog1
resnet_only__WBC_images_blog1['train_transforms'][-1] = transforms.Normalize(blood_WBC_baseline_mean, blood_WBC_baseline_std)
resnet_only__WBC_images_blog1['eval_transforms'][-1] = transforms.Normalize(blood_WBC_baseline_mean, blood_WBC_baseline_std)

vgg_covid19_only__WBC_images_blog1 = copy.deepcopy(resnet_only__WBC_images_blog1)
vgg_covid19_only__WBC_images_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(n_classes=4, learning_rate=learning_rate)'
vgg_covid19_only__WBC_images_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_covid19_only__WBC_images_blog1['traindir'] + "_" + "blog1"
vgg_covid19_only__WBC_images_blog1['train_transforms'].insert(len(vgg_covid19_only__WBC_images_blog1['train_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__WBC_images_blog1['eval_transforms'].insert(len(vgg_covid19_only__WBC_images_blog1['eval_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__WBC_images_blog1['train_transforms'][-1] = transforms.Normalize([0.7115], [0.0792])
vgg_covid19_only__WBC_images_blog1['eval_transforms'][-1] = transforms.Normalize([0.7115], [0.0792])

vgg19_pretrained_only__WBC_images_blog1 = copy.deepcopy(resnet_only__WBC_images_blog1)
vgg19_pretrained_only__WBC_images_blog1['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=5, learning_rate=learning_rate)'
vgg19_pretrained_only__WBC_images_blog1['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg19_pretrained_only__WBC_images_blog1['traindir'] + "_" + "blog1"

vgg16BN_pretrained_only__WBC_images_blog1 = copy.deepcopy(vgg19_pretrained_only__WBC_images_blog1)
vgg16BN_pretrained_only__WBC_images_blog1['model'] = 'Vgg16BNThreeChannelModelAllLayers(n_classes=5, learning_rate=learning_rate)'
vgg16BN_pretrained_only__WBC_images_blog1['name'] = 'Vgg16BNThreeChannelModelAllLayers' + "_" + vgg16BN_pretrained_only__WBC_images_blog1['traindir'] + "_" + "blog1"

resnet_untrained__WBC_images_blog1 = copy.deepcopy(resnet_only__WBC_images_blog1)
resnet_untrained__WBC_images_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate, pretrained=False)'
resnet_untrained__WBC_images_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_untrained__WBC_images_blog1['traindir'] + "_" + "blog1_untrained"

resnet_lower__WBC_images_blog1 = copy.deepcopy(resnet_only__WBC_images_blog1)
resnet_lower__WBC_images_blog1['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
resnet_lower__WBC_images_blog1['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__WBC_images_blog1['traindir'] + "_" + "blog1"

vgg_only__WBC_images_blog1 = copy.deepcopy(resnet_only__WBC_images_blog1)
vgg_only__WBC_images_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
vgg_only__WBC_images_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_only__WBC_images_blog1['traindir'] + "_" + "blog1"

vgg_untrained__WBC_images_blog1 = copy.deepcopy(vgg_only__WBC_images_blog1)
vgg_untrained__WBC_images_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate, pretrained=False)'
vgg_untrained__WBC_images_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_untrained__WBC_images_blog1['traindir'] + "_" + "blog1_untrained"

vgg_lower__WBC_images_blog1 = copy.deepcopy(resnet_only__WBC_images_blog1)
vgg_lower__WBC_images_blog1['model'] = 'VggModelLowerLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
vgg_lower__WBC_images_blog1['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__WBC_images_blog1['traindir'] + "_" + "blog1"

cnn_only__WBC_images_blog1 = copy.deepcopy(resnet_only__WBC_images_blog1)
cnn_only__WBC_images_blog1['model'] = 'CNNModel(n_classes=4, learning_rate=learning_rate)'
cnn_only__WBC_images_blog1['name'] = 'CNNModel' + "_" + cnn_only__WBC_images_blog1['traindir'] + "_" + "blog1"

cnn_mini__WBC_images_blog1 = copy.deepcopy(resnet_only__WBC_images_blog1)
cnn_mini__WBC_images_blog1['model'] = 'CNNMiniModel(n_classes=4, learning_rate=learning_rate)'
cnn_mini__WBC_images_blog1['name'] = 'CNNMiniModel' + "_" + cnn_mini__WBC_images_blog1['traindir'] + "_" + "blog1"

cnn_only__WBC_images_fullsize = copy.deepcopy(cnn_only__WBC_images_blog1)
cnn_only__WBC_images_fullsize['train_transforms'] = train_transforms_malaria_images_fullsize
cnn_only__WBC_images_fullsize['eval_transforms'] = eval_transforms_malaria_images_fullsize
cnn_only__WBC_images_fullsize['name'] = 'CNNModel' + "_" + cnn_only__WBC_images_fullsize['traindir'] + "_" + "fullsize"

cnn_only__WBC_images_tophat_otsu_thresh = copy.deepcopy(cnn_only__WBC_images_blog1)
cnn_only__WBC_images_tophat_otsu_thresh['name'] = 'CNNModel' + "_" + cnn_only__WBC_images_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
cnn_only__WBC_images_tophat_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=60))
cnn_only__WBC_images_tophat_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=60))
cnn_only__WBC_images_tophat_otsu_thresh['train_transforms'].insert(0, TopHat())
cnn_only__WBC_images_tophat_otsu_thresh['eval_transforms'].insert(0, TopHat())

cnn_only__WBC_images_tophat_mean_thresh = copy.deepcopy(cnn_only__WBC_images_blog1)
cnn_only__WBC_images_tophat_mean_thresh['name'] = 'CNNModel' + "_" + cnn_only__WBC_images_tophat_mean_thresh['traindir'] + "_" + "tophat_mean_thresh"
cnn_only__WBC_images_tophat_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=60))
cnn_only__WBC_images_tophat_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=60))
cnn_only__WBC_images_tophat_mean_thresh['train_transforms'].insert(0, TopHat())
cnn_only__WBC_images_tophat_mean_thresh['eval_transforms'].insert(0, TopHat())

cnn_only__WBC_images_sobel_otsu_thresh = copy.deepcopy(cnn_only__WBC_images_blog1)
cnn_only__WBC_images_sobel_otsu_thresh['name'] = 'CNNModel' + "_" + cnn_only__WBC_images_sobel_otsu_thresh['traindir'] + "_" + "otsu_thresh_sobel"
cnn_only__WBC_images_sobel_otsu_thresh['train_transforms'].insert(0, SobelX())
cnn_only__WBC_images_sobel_otsu_thresh['eval_transforms'].insert(0, SobelX())
cnn_only__WBC_images_sobel_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=60))
cnn_only__WBC_images_sobel_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=60))

cnn_only__WBC_images_sobel_mean_thresh = copy.deepcopy(cnn_only__WBC_images_blog1)
cnn_only__WBC_images_sobel_mean_thresh['name'] = 'CNNModel' + "_" + cnn_only__WBC_images_sobel_mean_thresh['traindir'] + "_" + "mean_thresh_sobel"
cnn_only__WBC_images_sobel_mean_thresh['train_transforms'].insert(0, SobelX())
cnn_only__WBC_images_sobel_mean_thresh['eval_transforms'].insert(0, SobelX())
cnn_only__WBC_images_sobel_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=60))
cnn_only__WBC_images_sobel_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=60))

cnn_only__WBC_images_erode = copy.deepcopy(cnn_only__WBC_images_blog1)
cnn_only__WBC_images_erode['name'] = 'CNNModel' + "_" + cnn_only__WBC_images_erode['traindir'] + "_" + "erode"
cnn_only__WBC_images_erode['train_transforms'].insert(0, ErosionAndDilation())
cnn_only__WBC_images_erode['eval_transforms'].insert(0, ErosionAndDilation())

cnn_only__WBC_images_clahe = copy.deepcopy(cnn_only__WBC_images_blog1)
cnn_only__WBC_images_clahe['name'] = 'CNNModel' + "_" + cnn_only__WBC_images_clahe['traindir'] + "_" + "clahe"
cnn_only__WBC_images_clahe['train_transforms'].insert(0, ContrastThroughHistogram())
cnn_only__WBC_images_clahe['eval_transforms'].insert(0, ContrastThroughHistogram())

cnn_only__WBC_images_LBP = copy.deepcopy(cnn_only__WBC_images_blog1)
cnn_only__WBC_images_LBP['name'] = 'CNNModel' + "_" + cnn_only__WBC_images_LBP['traindir'] + "_" + "LBP"
cnn_only__WBC_images_LBP['train_transforms'].insert(0, LocalBinaryPattern())
cnn_only__WBC_images_LBP['eval_transforms'].insert(0, LocalBinaryPattern())

# #########################################################################################################################################################

cat_counts_baseline_mean = [0.2971, 0.3081, 0.3042]
cat_counts_baseline_std = [0.2778, 0.2874, 0.2818]

resnet_only__cat_counts_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_only__cat_counts_blog1['traindir'] = 'cat_images_border' 
resnet_only__cat_counts_blog1['testdir'] = 'cat_images_border'
resnet_only__cat_counts_blog1['labels'] = os.listdir(resnet_only__cat_counts_blog1['traindir'])
resnet_only__cat_counts_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
resnet_only__cat_counts_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__cat_counts_blog1['traindir'] + "_" + "blog1"
resnet_only__cat_counts_blog1['train_transforms'] = train_transforms_malaria_images_blog1
resnet_only__cat_counts_blog1['eval_transforms'] = eval_transforms_malaria_images_blog1
resnet_only__cat_counts_blog1['train_transforms'][-1] = transforms.Normalize(cat_counts_baseline_mean, cat_counts_baseline_std)
resnet_only__cat_counts_blog1['eval_transforms'][-1] = transforms.Normalize(cat_counts_baseline_mean, cat_counts_baseline_std)

vgg_covid19_only__cat_counts_blog1 = copy.deepcopy(resnet_only__cat_counts_blog1)
vgg_covid19_only__cat_counts_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(n_classes=5, learning_rate=learning_rate)'
vgg_covid19_only__cat_counts_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_covid19_only__cat_counts_blog1['traindir'] + "_" + "blog1"
vgg_covid19_only__cat_counts_blog1['train_transforms'].insert(len(vgg_covid19_only__cat_counts_blog1['train_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__cat_counts_blog1['eval_transforms'].insert(len(vgg_covid19_only__cat_counts_blog1['eval_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__cat_counts_blog1['train_transforms'][-1] = transforms.Normalize([0.3044], [0.2687])
vgg_covid19_only__cat_counts_blog1['eval_transforms'][-1] = transforms.Normalize([0.3044], [0.2687])

vgg19_pretrained_only__cat_counts_blog1 = copy.deepcopy(resnet_only__cat_counts_blog1)
vgg19_pretrained_only__cat_counts_blog1['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=5, learning_rate=learning_rate)'
vgg19_pretrained_only__cat_counts_blog1['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg19_pretrained_only__cat_counts_blog1['traindir'] + "_" + "blog1"

vgg16BN_pretrained_only__cat_counts_blog1 = copy.deepcopy(vgg19_pretrained_only__cat_counts_blog1)
vgg16BN_pretrained_only__cat_counts_blog1['model'] = 'Vgg16BNThreeChannelModelAllLayers(n_classes=5, learning_rate=learning_rate)'
vgg16BN_pretrained_only__cat_counts_blog1['name'] = 'Vgg16BNThreeChannelModelAllLayers' + "_" + vgg16BN_pretrained_only__cat_counts_blog1['traindir'] + "_" + "blog1"

resnet_lower__cat_counts_blog1 = copy.deepcopy(resnet_only__cat_counts_blog1)
resnet_lower__cat_counts_blog1['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
resnet_lower__cat_counts_blog1['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__cat_counts_blog1['traindir'] + "_" + "blog1"

resnet_untrained__cat_counts_blog1 = copy.deepcopy(resnet_only__cat_counts_blog1)
resnet_untrained__cat_counts_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate, pretrained=False)'
resnet_untrained__cat_counts_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_untrained__cat_counts_blog1['traindir'] + "_" + "blog1_untrained"

vgg_only__cat_counts_blog1 = copy.deepcopy(resnet_only__cat_counts_blog1)
vgg_only__cat_counts_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
vgg_only__cat_counts_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_only__cat_counts_blog1['traindir'] + "_" + "blog1"

vgg_lower__cat_counts_blog1 = copy.deepcopy(resnet_only__cat_counts_blog1)
vgg_lower__cat_counts_blog1['model'] = 'VggModelLowerLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
vgg_lower__cat_counts_blog1['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__cat_counts_blog1['traindir'] + "_" + "blog1"

vgg_untrained__cat_counts_blog1 = copy.deepcopy(vgg_only__cat_counts_blog1)
vgg_untrained__cat_counts_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate, pretrained=False)'
vgg_untrained__cat_counts_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_untrained__cat_counts_blog1['traindir'] + "_" + "blog1_untrained"

cnn_only__cat_counts_blog1 = copy.deepcopy(resnet_only__cat_counts_blog1)
cnn_only__cat_counts_blog1['model'] = 'CNNModel(n_classes=5, learning_rate=learning_rate)'
cnn_only__cat_counts_blog1['name'] = 'CNNModel' + "_" + cnn_only__cat_counts_blog1['traindir'] + "_" + "blog1"

cnn_mini__cat_counts_blog1 = copy.deepcopy(resnet_only__cat_counts_blog1)
cnn_mini__cat_counts_blog1['model'] = 'CNNMiniModel(n_classes=5, learning_rate=learning_rate)'
cnn_mini__cat_counts_blog1['name'] = 'CNNMiniModel' + "_" + cnn_mini__cat_counts_blog1['traindir'] + "_" + "blog1"

cnn_only__cat_counts_fullsize = copy.deepcopy(cnn_only__cat_counts_blog1)
cnn_only__cat_counts_fullsize['model'] = 'CNNModel(n_classes=5, learning_rate=learning_rate, neurons=8192)'
cnn_only__cat_counts_fullsize['train_transforms'] = train_transforms_malaria_images_fullsize
cnn_only__cat_counts_fullsize['eval_transforms'] = eval_transforms_malaria_images_fullsize
cnn_only__cat_counts_fullsize['name'] = 'CNNModel' + "_" + cnn_only__cat_counts_fullsize['traindir'] + "_" + "fullsize"

vgg_only__cat_counts_tophat_otsu_thresh = copy.deepcopy(vgg_only__cat_counts_blog1)
vgg_only__cat_counts_tophat_otsu_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__cat_counts_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
vgg_only__cat_counts_tophat_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))
vgg_only__cat_counts_tophat_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))
vgg_only__cat_counts_tophat_otsu_thresh['train_transforms'].insert(0, TopHat())
vgg_only__cat_counts_tophat_otsu_thresh['eval_transforms'].insert(0, TopHat())

vgg_only__cat_counts_tophat_mean_thresh = copy.deepcopy(vgg_only__cat_counts_blog1)
vgg_only__cat_counts_tophat_mean_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__cat_counts_tophat_mean_thresh['traindir'] + "_" + "tophat_mean_thresh"
vgg_only__cat_counts_tophat_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))
vgg_only__cat_counts_tophat_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))
vgg_only__cat_counts_tophat_mean_thresh['train_transforms'].insert(0, TopHat())
vgg_only__cat_counts_tophat_mean_thresh['eval_transforms'].insert(0, TopHat())

vgg_only__cat_counts_sobel_otsu_thresh = copy.deepcopy(vgg_only__cat_counts_blog1)
vgg_only__cat_counts_sobel_otsu_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__cat_counts_sobel_otsu_thresh['traindir'] + "_" + "otsu_thresh_sobel"
vgg_only__cat_counts_sobel_otsu_thresh['train_transforms'].insert(0, SobelX())
vgg_only__cat_counts_sobel_otsu_thresh['eval_transforms'].insert(0, SobelX())
vgg_only__cat_counts_sobel_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))
vgg_only__cat_counts_sobel_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))

vgg_only__cat_counts_sobel_mean_thresh = copy.deepcopy(vgg_only__cat_counts_blog1)
vgg_only__cat_counts_sobel_mean_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__cat_counts_sobel_mean_thresh['traindir'] + "_" + "mean_thresh_sobel"
vgg_only__cat_counts_sobel_mean_thresh['train_transforms'].insert(0, SobelX())
vgg_only__cat_counts_sobel_mean_thresh['eval_transforms'].insert(0, SobelX())
vgg_only__cat_counts_sobel_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))
vgg_only__cat_counts_sobel_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))

vgg_only__cat_counts_erode = copy.deepcopy(vgg_only__cat_counts_blog1)
vgg_only__cat_counts_erode['name'] = 'VggModelAllLayers' + "_" + vgg_only__cat_counts_erode['traindir'] + "_" + "erode"
vgg_only__cat_counts_erode['train_transforms'].insert(0, ErosionAndDilation())
vgg_only__cat_counts_erode['eval_transforms'].insert(0, ErosionAndDilation())

vgg_only__cat_counts_clahe = copy.deepcopy(vgg_only__cat_counts_blog1)
vgg_only__cat_counts_clahe['name'] = 'VggModelAllLayers' + "_" + vgg_only__cat_counts_clahe['traindir'] + "_" + "clahe"
vgg_only__cat_counts_clahe['train_transforms'].insert(0, ContrastThroughHistogram())
vgg_only__cat_counts_clahe['eval_transforms'].insert(0, ContrastThroughHistogram())

vgg_only__cat_counts_LBP = copy.deepcopy(vgg_only__cat_counts_blog1)
vgg_only__cat_counts_LBP['name'] = 'VggModelAllLayers' + "_" + vgg_only__cat_counts_LBP['traindir'] + "_" + "LBP"
vgg_only__cat_counts_LBP['train_transforms'].insert(0, LocalBinaryPattern())
vgg_only__cat_counts_LBP['eval_transforms'].insert(0, LocalBinaryPattern())

# #########################################################################################################################################################

labeled_confocal_protein_baseline_mean = [0.2879, 0.2879, 0.2879]
labeled_confocal_protein_baseline_std = [0.2132, 0.2132, 0.2132]

resnet_only__labeled_confocal_protein_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_only__labeled_confocal_protein_blog1['traindir'] = 'labeled_confocal_protein' 
resnet_only__labeled_confocal_protein_blog1['testdir'] = 'labeled_confocal_protein'
resnet_only__labeled_confocal_protein_blog1['labels'] = os.listdir(resnet_only__labeled_confocal_protein_blog1['traindir'])
resnet_only__labeled_confocal_protein_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=9, learning_rate=learning_rate)'
resnet_only__labeled_confocal_protein_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1"
resnet_only__labeled_confocal_protein_blog1['train_transforms'] = train_transforms_malaria_images_blog1
resnet_only__labeled_confocal_protein_blog1['eval_transforms'] = eval_transforms_malaria_images_blog1
resnet_only__labeled_confocal_protein_blog1['train_transforms'][-1] = transforms.Normalize(labeled_confocal_protein_baseline_mean, labeled_confocal_protein_baseline_std)
resnet_only__labeled_confocal_protein_blog1['eval_transforms'][-1] = transforms.Normalize(labeled_confocal_protein_baseline_mean, labeled_confocal_protein_baseline_std)

vgg_covid19_only__labeled_confocal_protein_blog1 = copy.deepcopy(resnet_only__labeled_confocal_protein_blog1)
vgg_covid19_only__labeled_confocal_protein_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(n_classes=9, learning_rate=learning_rate)'
vgg_covid19_only__labeled_confocal_protein_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_covid19_only__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1"
vgg_covid19_only__labeled_confocal_protein_blog1['train_transforms'].insert(len(vgg_covid19_only__labeled_confocal_protein_blog1['train_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__labeled_confocal_protein_blog1['eval_transforms'].insert(len(vgg_covid19_only__labeled_confocal_protein_blog1['eval_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__labeled_confocal_protein_blog1['train_transforms'][-1] = transforms.Normalize([0.2879], [0.2132])
vgg_covid19_only__labeled_confocal_protein_blog1['eval_transforms'][-1] = transforms.Normalize([0.2879], [0.2132])

vgg19_pretrained_only__labeled_confocal_protein_blog1 = copy.deepcopy(resnet_only__labeled_confocal_protein_blog1)
vgg19_pretrained_only__labeled_confocal_protein_blog1['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=9, learning_rate=learning_rate)'
vgg19_pretrained_only__labeled_confocal_protein_blog1['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg19_pretrained_only__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1"

vgg16BN_pretrained_only__labeled_confocal_protein_blog1 = copy.deepcopy(vgg19_pretrained_only__labeled_confocal_protein_blog1)
vgg16BN_pretrained_only__labeled_confocal_protein_blog1['model'] = 'Vgg16BNThreeChannelModelAllLayers(n_classes=9, learning_rate=learning_rate)'
vgg16BN_pretrained_only__labeled_confocal_protein_blog1['name'] = 'Vgg16BNThreeChannelModelAllLayers' + "_" + vgg16BN_pretrained_only__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1"

resnet_lower__labeled_confocal_protein_blog1 = copy.deepcopy(resnet_only__labeled_confocal_protein_blog1)
resnet_lower__labeled_confocal_protein_blog1['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=9, learning_rate=learning_rate)'
resnet_lower__labeled_confocal_protein_blog1['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1"

resnet_untrained__labeled_confocal_protein_blog1 = copy.deepcopy(resnet_only__labeled_confocal_protein_blog1)
resnet_untrained__labeled_confocal_protein_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=9, learning_rate=learning_rate, pretrained=False)'
resnet_untrained__labeled_confocal_protein_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_untrained__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1_untrained"

vgg_only__labeled_confocal_protein_blog1 = copy.deepcopy(resnet_only__labeled_confocal_protein_blog1)
vgg_only__labeled_confocal_protein_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=9, learning_rate=learning_rate)'
vgg_only__labeled_confocal_protein_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_only__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1"

vgg_lower__labeled_confocal_protein_blog1 = copy.deepcopy(resnet_only__labeled_confocal_protein_blog1)
vgg_lower__labeled_confocal_protein_blog1['model'] = 'VggModelLowerLayers(freeze=False, n_classes=9, learning_rate=learning_rate)'
vgg_lower__labeled_confocal_protein_blog1['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1"

vgg_untrained__labeled_confocal_protein_blog1 = copy.deepcopy(vgg_only__labeled_confocal_protein_blog1)
vgg_untrained__labeled_confocal_protein_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=9, learning_rate=learning_rate, pretrained=False)'
vgg_untrained__labeled_confocal_protein_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_untrained__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1_untrained"

cnn_only__labeled_confocal_protein_blog1 = copy.deepcopy(resnet_only__labeled_confocal_protein_blog1)
cnn_only__labeled_confocal_protein_blog1['model'] = 'CNNModel(n_classes=9, learning_rate=learning_rate)'
cnn_only__labeled_confocal_protein_blog1['name'] = 'CNNModel' + "_" + cnn_only__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1"

cnn_mini__labeled_confocal_protein_blog1 = copy.deepcopy(resnet_only__labeled_confocal_protein_blog1)
cnn_mini__labeled_confocal_protein_blog1['model'] = 'CNNMiniModel(n_classes=9, learning_rate=learning_rate)'
cnn_mini__labeled_confocal_protein_blog1['name'] = 'CNNMiniModel' + "_" + cnn_mini__labeled_confocal_protein_blog1['traindir'] + "_" + "blog1"

cnn_only__labeled_confocal_protein_halfsize = copy.deepcopy(cnn_only__labeled_confocal_protein_blog1)
cnn_only__labeled_confocal_protein_halfsize['model'] = 'CNNModel(n_classes=9, learning_rate=learning_rate, neurons=51200)'
cnn_only__labeled_confocal_protein_halfsize['train_transforms'] = train_transforms_malaria_images_halfsize
cnn_only__labeled_confocal_protein_halfsize['eval_transforms'] = eval_transforms_malaria_images_halfsize
cnn_only__labeled_confocal_protein_halfsize['name'] = 'CNNModel' + "_" + cnn_only__labeled_confocal_protein_halfsize['traindir'] + "_" + "fullsize"

resnet_lower__labeled_confocal_protein_tophat_otsu_thresh = copy.deepcopy(resnet_lower__labeled_confocal_protein_blog1)
resnet_lower__labeled_confocal_protein_tophat_otsu_thresh['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__labeled_confocal_protein_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
resnet_lower__labeled_confocal_protein_tophat_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))
resnet_lower__labeled_confocal_protein_tophat_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))
resnet_lower__labeled_confocal_protein_tophat_otsu_thresh['train_transforms'].insert(0, TopHat())
resnet_lower__labeled_confocal_protein_tophat_otsu_thresh['eval_transforms'].insert(0, TopHat())

resnet_lower__labeled_confocal_protein_tophat_mean_thresh = copy.deepcopy(resnet_lower__labeled_confocal_protein_blog1)
resnet_lower__labeled_confocal_protein_tophat_mean_thresh['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__labeled_confocal_protein_tophat_mean_thresh['traindir'] + "_" + "tophat_mean_thresh"
resnet_lower__labeled_confocal_protein_tophat_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))
resnet_lower__labeled_confocal_protein_tophat_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))
resnet_lower__labeled_confocal_protein_tophat_mean_thresh['train_transforms'].insert(0, TopHat())
resnet_lower__labeled_confocal_protein_tophat_mean_thresh['eval_transforms'].insert(0, TopHat())

resnet_lower__labeled_confocal_protein_sobel_otsu_thresh = copy.deepcopy(resnet_lower__labeled_confocal_protein_blog1)
resnet_lower__labeled_confocal_protein_sobel_otsu_thresh['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__labeled_confocal_protein_sobel_otsu_thresh['traindir'] + "_" + "otsu_thresh_sobel"
resnet_lower__labeled_confocal_protein_sobel_otsu_thresh['train_transforms'].insert(0, SobelX())
resnet_lower__labeled_confocal_protein_sobel_otsu_thresh['eval_transforms'].insert(0, SobelX())
resnet_lower__labeled_confocal_protein_sobel_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))
resnet_lower__labeled_confocal_protein_sobel_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=5))

resnet_lower__labeled_confocal_protein_sobel_mean_thresh = copy.deepcopy(resnet_lower__labeled_confocal_protein_blog1)
resnet_lower__labeled_confocal_protein_sobel_mean_thresh['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__labeled_confocal_protein_sobel_mean_thresh['traindir'] + "_" + "mean_thresh_sobel"
resnet_lower__labeled_confocal_protein_sobel_mean_thresh['train_transforms'].insert(0, SobelX())
resnet_lower__labeled_confocal_protein_sobel_mean_thresh['eval_transforms'].insert(0, SobelX())
resnet_lower__labeled_confocal_protein_sobel_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))
resnet_lower__labeled_confocal_protein_sobel_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=5))

resnet_lower__labeled_confocal_protein_erode = copy.deepcopy(resnet_lower__labeled_confocal_protein_blog1)
resnet_lower__labeled_confocal_protein_erode['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__labeled_confocal_protein_erode['traindir'] + "_" + "erode"
resnet_lower__labeled_confocal_protein_erode['train_transforms'].insert(0, ErosionAndDilation())
resnet_lower__labeled_confocal_protein_erode['eval_transforms'].insert(0, ErosionAndDilation())

resnet_lower__labeled_confocal_protein_clahe = copy.deepcopy(resnet_lower__labeled_confocal_protein_blog1)
resnet_lower__labeled_confocal_protein_clahe['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__labeled_confocal_protein_clahe['traindir'] + "_" + "clahe"
resnet_lower__labeled_confocal_protein_clahe['train_transforms'].insert(0, ContrastThroughHistogram())
resnet_lower__labeled_confocal_protein_clahe['eval_transforms'].insert(0, ContrastThroughHistogram())

cnn_only__labeled_confocal_protein_LBP = copy.deepcopy(cnn_only__labeled_confocal_protein_blog1)
cnn_only__labeled_confocal_protein_LBP['name'] = 'CNNModel' + "_" + cnn_only__labeled_confocal_protein_LBP['traindir'] + "_" + "LBP"
cnn_only__labeled_confocal_protein_LBP['train_transforms'].insert(0, LocalBinaryPattern())
cnn_only__labeled_confocal_protein_LBP['eval_transforms'].insert(0, LocalBinaryPattern())

resnet_only__labeled_confocal_protein_blog1_20e = copy.deepcopy(resnet_only__labeled_confocal_protein_blog1)
resnet_only__labeled_confocal_protein_blog1_20e['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__labeled_confocal_protein_blog1_20e['traindir'] + "_" + "blog1_20e"
resnet_only__labeled_confocal_protein_blog1_20e['epochs'] = 20

cnn_only__labeled_confocal_protein__blog1_venn = copy.deepcopy(cnn_only__labeled_confocal_protein_blog1)
cnn_only__labeled_confocal_protein__blog1_venn['name'] = 'CNNModel' + "_" + cnn_only__labeled_confocal_protein__blog1_venn['traindir'] + "_" + "blog1_venn"
cnn_only__labeled_confocal_protein__blog1_venn['venn_data'] = './CNNModel_labeled_confocal_protein_blog1_venn_data.results21_5.csv'

# #########################################################################################################################################################
CHO_images_baseline_mean = [0.0514, 0.0514, 0.0514]
CHO_images_baseline_std = [0.0446, 0.0446, 0.0446]

resnet_only__CHO_images_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_only__CHO_images_blog1['traindir'] = 'CHO_images' 
resnet_only__CHO_images_blog1['testdir'] = 'CHO_images'
resnet_only__CHO_images_blog1['labels'] = os.listdir(resnet_only__CHO_images_blog1['traindir'])
resnet_only__CHO_images_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
resnet_only__CHO_images_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__CHO_images_blog1['traindir'] + "_" + "blog1"
resnet_only__CHO_images_blog1['train_transforms'] = train_transforms_malaria_images_blog1
resnet_only__CHO_images_blog1['eval_transforms'] = eval_transforms_malaria_images_blog1
resnet_only__CHO_images_blog1['train_transforms'][-1] = transforms.Normalize(CHO_images_baseline_mean, CHO_images_baseline_std)
resnet_only__CHO_images_blog1['eval_transforms'][-1] = transforms.Normalize(CHO_images_baseline_mean, CHO_images_baseline_std)

vgg_covid19_only__CHO_images_blog1 = copy.deepcopy(resnet_only__CHO_images_blog1)
vgg_covid19_only__CHO_images_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(n_classes=5, learning_rate=learning_rate)'
vgg_covid19_only__CHO_images_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_covid19_only__CHO_images_blog1['traindir'] + "_" + "blog1"
vgg_covid19_only__CHO_images_blog1['train_transforms'].insert(len(vgg_covid19_only__CHO_images_blog1['train_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__CHO_images_blog1['eval_transforms'].insert(len(vgg_covid19_only__CHO_images_blog1['eval_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__CHO_images_blog1['train_transforms'][-1] = transforms.Normalize([0.0514], [0.0446])
vgg_covid19_only__CHO_images_blog1['eval_transforms'][-1] = transforms.Normalize([0.0514], [0.0446])

vgg19_pretrained_only__CHO_images_blog1 = copy.deepcopy(resnet_only__CHO_images_blog1)
vgg19_pretrained_only__CHO_images_blog1['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=5, learning_rate=learning_rate)'
vgg19_pretrained_only__CHO_images_blog1['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg19_pretrained_only__CHO_images_blog1['traindir'] + "_" + "blog1"

vgg16BN_pretrained_only__CHO_images_blog1 = copy.deepcopy(vgg19_pretrained_only__CHO_images_blog1)
vgg16BN_pretrained_only__CHO_images_blog1['model'] = 'Vgg16BNThreeChannelModelAllLayers(n_classes=5, learning_rate=learning_rate)'
vgg16BN_pretrained_only__CHO_images_blog1['name'] = 'Vgg16BNThreeChannelModelAllLayers' + "_" + vgg16BN_pretrained_only__CHO_images_blog1['traindir'] + "_" + "blog1"

resnet_lower__CHO_images_blog1 = copy.deepcopy(resnet_only__CHO_images_blog1)
resnet_lower__CHO_images_blog1['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
resnet_lower__CHO_images_blog1['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__CHO_images_blog1['traindir'] + "_" + "blog1"

resnet_untrained__CHO_images_blog1 = copy.deepcopy(resnet_only__CHO_images_blog1)
resnet_untrained__CHO_images_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate, pretrained=False)'
resnet_untrained__CHO_images_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_untrained__CHO_images_blog1['traindir'] + "_" + "blog1_untrained"

vgg_only__CHO_images_blog1 = copy.deepcopy(resnet_only__CHO_images_blog1)
vgg_only__CHO_images_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
vgg_only__CHO_images_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_only__CHO_images_blog1['traindir'] + "_" + "blog1"

vgg_lower__CHO_images_blog1 = copy.deepcopy(resnet_only__CHO_images_blog1)
vgg_lower__CHO_images_blog1['model'] = 'VggModelLowerLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
vgg_lower__CHO_images_blog1['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__CHO_images_blog1['traindir'] + "_" + "blog1"

vgg_untrained__CHO_images_blog1 = copy.deepcopy(vgg_only__CHO_images_blog1)
vgg_untrained__CHO_images_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate, pretrained=False)'
vgg_untrained__CHO_images_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_untrained__CHO_images_blog1['traindir'] + "_" + "blog1_untrained"

cnn_only__CHO_images_blog1 = copy.deepcopy(resnet_only__CHO_images_blog1)
cnn_only__CHO_images_blog1['model'] = 'CNNModel(n_classes=5, learning_rate=learning_rate)'
cnn_only__CHO_images_blog1['name'] = 'CNNModel' + "_" + cnn_only__CHO_images_blog1['traindir'] + "_" + "blog1"

cnn_mini__CHO_images_blog1 = copy.deepcopy(resnet_only__CHO_images_blog1)
cnn_mini__CHO_images_blog1['model'] = 'CNNMiniModel(n_classes=5, learning_rate=learning_rate)'
cnn_mini__CHO_images_blog1['name'] = 'CNNMiniModel' + "_" + cnn_mini__CHO_images_blog1['traindir'] + "_" + "blog1"

cnn_only__CHO_images_fullsize = copy.deepcopy(cnn_only__CHO_images_blog1)
cnn_only__CHO_images_fullsize['model'] = 'CNNModel(n_classes=9, learning_rate=learning_rate, neurons=373248)'
cnn_only__CHO_images_fullsize['train_transforms'] = train_transforms_malaria_images_fullsize
cnn_only__CHO_images_fullsize['eval_transforms'] = eval_transforms_malaria_images_fullsize
cnn_only__CHO_images_fullsize['name'] = 'CNNModel' + "_" + cnn_only__CHO_images_fullsize['traindir'] + "_" + "fullsize"

vgg_only__CHO_images_tophat_otsu_thresh = copy.deepcopy(vgg_only__CHO_images_blog1)
vgg_only__CHO_images_tophat_otsu_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__CHO_images_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
vgg_only__CHO_images_tophat_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=50))
vgg_only__CHO_images_tophat_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=50))
vgg_only__CHO_images_tophat_otsu_thresh['train_transforms'].insert(0, TopHat())
vgg_only__CHO_images_tophat_otsu_thresh['eval_transforms'].insert(0, TopHat())

vgg_only__CHO_images_tophat_mean_thresh = copy.deepcopy(vgg_only__CHO_images_blog1)
vgg_only__CHO_images_tophat_mean_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__CHO_images_tophat_mean_thresh['traindir'] + "_" + "tophat_mean_thresh"
vgg_only__CHO_images_tophat_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=50))
vgg_only__CHO_images_tophat_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=50))
vgg_only__CHO_images_tophat_mean_thresh['train_transforms'].insert(0, TopHat())
vgg_only__CHO_images_tophat_mean_thresh['eval_transforms'].insert(0, TopHat())

vgg_only__CHO_images_sobel_otsu_thresh = copy.deepcopy(vgg_only__CHO_images_blog1)
vgg_only__CHO_images_sobel_otsu_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__CHO_images_sobel_otsu_thresh['traindir'] + "_" + "otsu_thresh_sobel"
vgg_only__CHO_images_sobel_otsu_thresh['train_transforms'].insert(0, SobelX())
vgg_only__CHO_images_sobel_otsu_thresh['eval_transforms'].insert(0, SobelX())
vgg_only__CHO_images_sobel_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=50))
vgg_only__CHO_images_sobel_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=50))

vgg_only__CHO_images_sobel_mean_thresh = copy.deepcopy(vgg_only__CHO_images_blog1)
vgg_only__CHO_images_sobel_mean_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__CHO_images_sobel_mean_thresh['traindir'] + "_" + "mean_thresh_sobel"
vgg_only__CHO_images_sobel_mean_thresh['train_transforms'].insert(0, SobelX())
vgg_only__CHO_images_sobel_mean_thresh['eval_transforms'].insert(0, SobelX())
vgg_only__CHO_images_sobel_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=50))
vgg_only__CHO_images_sobel_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=50))

vgg_only__CHO_images_erode = copy.deepcopy(vgg_only__CHO_images_blog1)
vgg_only__CHO_images_erode['name'] = 'VggModelAllLayers' + "_" + vgg_only__CHO_images_erode['traindir'] + "_" + "erode"
vgg_only__CHO_images_erode['train_transforms'].insert(0, ErosionAndDilation())
vgg_only__CHO_images_erode['eval_transforms'].insert(0, ErosionAndDilation())

vgg_only__CHO_images_clahe = copy.deepcopy(vgg_only__CHO_images_blog1)
vgg_only__CHO_images_clahe['name'] = 'VggModelAllLayers' + "_" + vgg_only__CHO_images_clahe['traindir'] + "_" + "clahe"
vgg_only__CHO_images_clahe['train_transforms'].insert(0, ContrastThroughHistogram())
vgg_only__CHO_images_clahe['eval_transforms'].insert(0, ContrastThroughHistogram())

cnn_only__CHO_images_LBP = copy.deepcopy(cnn_only__CHO_images_blog1)
cnn_only__CHO_images_LBP['name'] = 'CNNModel' + "_" + cnn_only__CHO_images_LBP['traindir'] + "_" + "LBP"
cnn_only__CHO_images_LBP['train_transforms'].insert(0, LocalBinaryPattern())
cnn_only__CHO_images_LBP['eval_transforms'].insert(0, LocalBinaryPattern())

resnet_only__CHO_images_blog1_20e = copy.deepcopy(resnet_only__CHO_images_blog1)
resnet_only__CHO_images_blog1_20e['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__CHO_images_blog1_20e['traindir'] + "_" + "blog1_20e"
resnet_only__CHO_images_blog1_20e['epochs'] = 20

cnn_only__CHO_images__blog1_venn = copy.deepcopy(cnn_only__CHO_images_blog1)
cnn_only__CHO_images__blog1_venn['name'] = 'CNNModel' + "_" + cnn_only__CHO_images__blog1_venn['traindir'] + "_" + "blog1_venn"
cnn_only__CHO_images__blog1_venn['venn_data'] = './CNNModel_CHO_images_blog1_venn_data.results99_9.csv'

# #########################################################################################################################################################
synimages_baseline_mean = [0.0427, 0.0058, 0.0147]
synimages_baseline_std = [0.1487, 0.0339, 0.1005]

resnet_only__synimages_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_only__synimages_blog1['traindir'] = 'synimages' 
resnet_only__synimages_blog1['testdir'] = 'synimages'
resnet_only__synimages_blog1['labels'] = os.listdir(resnet_only__synimages_blog1['traindir'])
resnet_only__synimages_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=6, learning_rate=learning_rate)'
resnet_only__synimages_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__synimages_blog1['traindir'] + "_" + "blog1"
resnet_only__synimages_blog1['train_transforms'] = train_transforms_malaria_images_blog1
resnet_only__synimages_blog1['eval_transforms'] = eval_transforms_malaria_images_blog1
resnet_only__synimages_blog1['train_transforms'][-1] = transforms.Normalize(synimages_baseline_mean, synimages_baseline_std)
resnet_only__synimages_blog1['eval_transforms'][-1] = transforms.Normalize(synimages_baseline_mean, synimages_baseline_std)

vgg_covid19_only__synimages_blog1 = copy.deepcopy(resnet_only__synimages_blog1)
vgg_covid19_only__synimages_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(n_classes=6, learning_rate=learning_rate)'
vgg_covid19_only__synimages_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_covid19_only__synimages_blog1['traindir'] + "_" + "blog1"
vgg_covid19_only__synimages_blog1['train_transforms'].insert(len(vgg_covid19_only__synimages_blog1['train_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__synimages_blog1['eval_transforms'].insert(len(vgg_covid19_only__synimages_blog1['eval_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__synimages_blog1['train_transforms'][-1] = transforms.Normalize([0.0178], [0.0534])
vgg_covid19_only__synimages_blog1['eval_transforms'][-1] = transforms.Normalize([0.0178], [0.0534])

vgg19_pretrained_only__synimages_blog1 = copy.deepcopy(resnet_only__synimages_blog1)
vgg19_pretrained_only__synimages_blog1['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=6, learning_rate=learning_rate)'
vgg19_pretrained_only__synimages_blog1['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg19_pretrained_only__synimages_blog1['traindir'] + "_" + "blog1"

vgg16BN_pretrained_only__synimages_blog1 = copy.deepcopy(vgg19_pretrained_only__synimages_blog1)
vgg16BN_pretrained_only__synimages_blog1['model'] = 'Vgg16BNThreeChannelModelAllLayers(n_classes=6, learning_rate=learning_rate)'
vgg16BN_pretrained_only__synimages_blog1['name'] = 'Vgg16BNThreeChannelModelAllLayers' + "_" + vgg16BN_pretrained_only__synimages_blog1['traindir'] + "_" + "blog1"

resnet_lower__synimages_blog1 = copy.deepcopy(resnet_only__synimages_blog1)
resnet_lower__synimages_blog1['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=6, learning_rate=learning_rate)'
resnet_lower__synimages_blog1['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__synimages_blog1['traindir'] + "_" + "blog1"

resnet_untrained__synimages_blog1 = copy.deepcopy(resnet_only__synimages_blog1)
resnet_untrained__synimages_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=6, learning_rate=learning_rate, pretrained=False)'
resnet_untrained__synimages_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_untrained__synimages_blog1['traindir'] + "_" + "blog1_untrained"

vgg_only__synimages_blog1 = copy.deepcopy(resnet_only__synimages_blog1)
vgg_only__synimages_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=6, learning_rate=learning_rate)'
vgg_only__synimages_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_only__synimages_blog1['traindir'] + "_" + "blog1"

vgg_lower__synimages_blog1 = copy.deepcopy(resnet_only__synimages_blog1)
vgg_lower__synimages_blog1['model'] = 'VggModelLowerLayers(freeze=False, n_classes=6, learning_rate=learning_rate)'
vgg_lower__synimages_blog1['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__synimages_blog1['traindir'] + "_" + "blog1"

vgg_untrained__synimages_blog1 = copy.deepcopy(vgg_only__synimages_blog1)
vgg_untrained__synimages_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=6, learning_rate=learning_rate, pretrained=False)'
vgg_untrained__synimages_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_untrained__synimages_blog1['traindir'] + "_" + "blog1_untrained"

cnn_only__synimages_blog1 = copy.deepcopy(resnet_only__synimages_blog1)
cnn_only__synimages_blog1['model'] = 'CNNModel(n_classes=6, learning_rate=learning_rate)'
cnn_only__synimages_blog1['name'] = 'CNNModel' + "_" + cnn_only__synimages_blog1['traindir'] + "_" + "blog1"

cnn_mini__synimages_blog1 = copy.deepcopy(resnet_only__synimages_blog1)
cnn_mini__synimages_blog1['model'] = 'CNNMiniModel(n_classes=6, learning_rate=learning_rate)'
cnn_mini__synimages_blog1['name'] = 'CNNMiniModel' + "_" + cnn_mini__synimages_blog1['traindir'] + "_" + "blog1"

vgg_lower__synimages_centercrop = copy.deepcopy(vgg_lower__synimages_blog1)
vgg_lower__synimages_centercrop['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__synimages_centercrop['traindir'] + "_" + "centercrop"
vgg_lower__synimages_centercrop['train_transforms'][4] = transforms.CenterCrop(size=[224,224])
vgg_lower__synimages_centercrop['eval_transforms'][0] = transforms.CenterCrop(size=[224,224])

cnn_only__synimages_fullsize = copy.deepcopy(cnn_only__synimages_blog1)
cnn_only__synimages_fullsize['train_transforms'] = train_transforms_malaria_images_fullsize
cnn_only__synimages_fullsize['eval_transforms'] = eval_transforms_malaria_images_fullsize
cnn_only__synimages_fullsize['name'] = 'CNNModel' + "_" + cnn_only__synimages_fullsize['traindir'] + "_" + "fullsize"

vgg_lower__synimages_tophat_otsu_thresh = copy.deepcopy(vgg_lower__synimages_blog1)
vgg_lower__synimages_tophat_otsu_thresh['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__synimages_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
vgg_lower__synimages_tophat_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=10))
vgg_lower__synimages_tophat_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=10))
vgg_lower__synimages_tophat_otsu_thresh['train_transforms'].insert(0, TopHat())
vgg_lower__synimages_tophat_otsu_thresh['eval_transforms'].insert(0, TopHat())

vgg_lower__synimages_tophat_mean_thresh = copy.deepcopy(vgg_lower__synimages_blog1)
vgg_lower__synimages_tophat_mean_thresh['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__synimages_tophat_mean_thresh['traindir'] + "_" + "tophat_mean_thresh"
vgg_lower__synimages_tophat_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=10))
vgg_lower__synimages_tophat_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=10))
vgg_lower__synimages_tophat_mean_thresh['train_transforms'].insert(0, TopHat())
vgg_lower__synimages_tophat_mean_thresh['eval_transforms'].insert(0, TopHat())

vgg_lower__synimages_sobel_otsu_thresh = copy.deepcopy(vgg_lower__synimages_blog1)
vgg_lower__synimages_sobel_otsu_thresh['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__synimages_sobel_otsu_thresh['traindir'] + "_" + "otsu_thresh_sobel"
vgg_lower__synimages_sobel_otsu_thresh['train_transforms'].insert(0, SobelX())
vgg_lower__synimages_sobel_otsu_thresh['eval_transforms'].insert(0, SobelX())
vgg_lower__synimages_sobel_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=10))
vgg_lower__synimages_sobel_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=10))

vgg_lower__synimages_sobel_mean_thresh = copy.deepcopy(vgg_lower__synimages_blog1)
vgg_lower__synimages_sobel_mean_thresh['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__synimages_sobel_mean_thresh['traindir'] + "_" + "mean_thresh_sobel"
vgg_lower__synimages_sobel_mean_thresh['train_transforms'].insert(0, SobelX())
vgg_lower__synimages_sobel_mean_thresh['eval_transforms'].insert(0, SobelX())
vgg_lower__synimages_sobel_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=10))
vgg_lower__synimages_sobel_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=10))

vgg_lower__synimages_erode = copy.deepcopy(vgg_lower__synimages_blog1)
vgg_lower__synimages_erode['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__synimages_erode['traindir'] + "_" + "erode"
vgg_lower__synimages_erode['train_transforms'].insert(0, ErosionAndDilation())
vgg_lower__synimages_erode['eval_transforms'].insert(0, ErosionAndDilation())

vgg_lower__synimages_clahe = copy.deepcopy(vgg_lower__synimages_blog1)
vgg_lower__synimages_clahe['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__synimages_clahe['traindir'] + "_" + "clahe"
vgg_lower__synimages_clahe['train_transforms'].insert(0, ContrastThroughHistogram())
vgg_lower__synimages_clahe['eval_transforms'].insert(0, ContrastThroughHistogram())

vgg_lower__synimages_LBP = copy.deepcopy(vgg_lower__synimages_blog1)
vgg_lower__synimages_LBP['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__synimages_LBP['traindir'] + "_" + "LBP"
vgg_lower__synimages_LBP['train_transforms'].insert(0, LocalBinaryPattern())
vgg_lower__synimages_LBP['eval_transforms'].insert(0, LocalBinaryPattern())

# #########################################################################################################################################################
covid_mini_baseline_mean = [0.0645, 0.0645, 0.0645]
covid_mini_baseline_std = [0.0498, 0.0498, 0.0498]

resnet_only__covid_mini_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_only__covid_mini_blog1['traindir'] = 'covid_mini' 
resnet_only__covid_mini_blog1['testdir'] = 'covid_mini'
resnet_only__covid_mini_blog1['labels'] = os.listdir(resnet_only__covid_mini_blog1['traindir'])
resnet_only__covid_mini_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
resnet_only__covid_mini_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__covid_mini_blog1['traindir'] + "_" + "blog1"
resnet_only__covid_mini_blog1['train_transforms'] = train_transforms_malaria_images_blog1
resnet_only__covid_mini_blog1['eval_transforms'] = eval_transforms_malaria_images_blog1
resnet_only__covid_mini_blog1['train_transforms'][-1] = transforms.Normalize(covid_mini_baseline_mean, covid_mini_baseline_std)
resnet_only__covid_mini_blog1['eval_transforms'][-1] = transforms.Normalize(covid_mini_baseline_mean, covid_mini_baseline_std)

resnet_lower__covid_mini_blog1 = copy.deepcopy(resnet_only__covid_mini_blog1)
resnet_lower__covid_mini_blog1['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
resnet_lower__covid_mini_blog1['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__covid_mini_blog1['traindir'] + "_" + "blog1"

resnet_untrained__covid_mini_blog1 = copy.deepcopy(resnet_only__covid_mini_blog1)
resnet_untrained__covid_mini_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate, pretrained=False)'
resnet_untrained__covid_mini_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_untrained__covid_mini_blog1['traindir'] + "_" + "blog1_untrained"

vgg_only__covid_mini_blog1 = copy.deepcopy(resnet_only__covid_mini_blog1)
vgg_only__covid_mini_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
vgg_only__covid_mini_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_only__covid_mini_blog1['traindir'] + "_" + "blog1"

vgg_lower__covid_mini_blog1 = copy.deepcopy(resnet_only__covid_mini_blog1)
vgg_lower__covid_mini_blog1['model'] = 'VggModelLowerLayers(freeze=False, n_classes=5, learning_rate=learning_rate)'
vgg_lower__covid_mini_blog1['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_mini_blog1['traindir'] + "_" + "blog1"

vgg_untrained__covid_mini_blog1 = copy.deepcopy(vgg_only__covid_mini_blog1)
vgg_untrained__covid_mini_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=5, learning_rate=learning_rate, pretrained=False)'
vgg_untrained__covid_mini_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_untrained__covid_mini_blog1['traindir'] + "_" + "blog1_untrained"

cnn_only__covid_mini_blog1 = copy.deepcopy(resnet_only__covid_mini_blog1)
cnn_only__covid_mini_blog1['model'] = 'CNNModel(n_classes=5, learning_rate=learning_rate)'
cnn_only__covid_mini_blog1['name'] = 'CNNModel' + "_" + cnn_only__covid_mini_blog1['traindir'] + "_" + "blog1"

cnn_mini__covid_mini_blog1 = copy.deepcopy(resnet_only__covid_mini_blog1)
cnn_mini__covid_mini_blog1['model'] = 'CNNMiniModel(n_classes=5, learning_rate=learning_rate)'
cnn_mini__covid_mini_blog1['name'] = 'CNNMiniModel' + "_" + cnn_mini__covid_mini_blog1['traindir'] + "_" + "blog1"

vgg_lower__covid_mini_centercrop = copy.deepcopy(vgg_lower__covid_mini_blog1)
vgg_lower__covid_mini_centercrop['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_mini_centercrop['traindir'] + "_" + "centercrop"
vgg_lower__covid_mini_centercrop['train_transforms'][4] = transforms.CenterCrop(size=[224,224])
vgg_lower__covid_mini_centercrop['eval_transforms'][0] = transforms.CenterCrop(size=[224,224])

cnn_only__covid_mini_fullsize = copy.deepcopy(cnn_only__covid_mini_blog1)
cnn_only__covid_mini_fullsize['model'] = 'CNNModel(n_classes=5, learning_rate=learning_rate, neurons=373248)'
cnn_only__covid_mini_fullsize['train_transforms'] = train_transforms_malaria_images_fullsize
cnn_only__covid_mini_fullsize['eval_transforms'] = eval_transforms_malaria_images_fullsize
cnn_only__covid_mini_fullsize['name'] = 'CNNModel' + "_" + cnn_only__covid_mini_fullsize['traindir'] + "_" + "fullsize"

vgg_lower__covid_mini_tophat_otsu_thresh = copy.deepcopy(vgg_lower__covid_mini_blog1)
vgg_lower__covid_mini_tophat_otsu_thresh['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_mini_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
vgg_lower__covid_mini_tophat_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=1))
vgg_lower__covid_mini_tophat_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=1))
vgg_lower__covid_mini_tophat_otsu_thresh['train_transforms'].insert(0, TopHat())
vgg_lower__covid_mini_tophat_otsu_thresh['eval_transforms'].insert(0, TopHat())

vgg_lower__covid_mini_tophat_mean_thresh = copy.deepcopy(vgg_lower__covid_mini_blog1)
vgg_lower__covid_mini_tophat_mean_thresh['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_mini_tophat_mean_thresh['traindir'] + "_" + "tophat_mean_thresh"
vgg_lower__covid_mini_tophat_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=1))
vgg_lower__covid_mini_tophat_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=1))
vgg_lower__covid_mini_tophat_mean_thresh['train_transforms'].insert(0, TopHat())
vgg_lower__covid_mini_tophat_mean_thresh['eval_transforms'].insert(0, TopHat())

vgg_lower__covid_mini_sobel_otsu_thresh = copy.deepcopy(vgg_lower__covid_mini_blog1)
vgg_lower__covid_mini_sobel_otsu_thresh['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_mini_sobel_otsu_thresh['traindir'] + "_" + "otsu_thresh_sobel"
vgg_lower__covid_mini_sobel_otsu_thresh['train_transforms'].insert(0, SobelX())
vgg_lower__covid_mini_sobel_otsu_thresh['eval_transforms'].insert(0, SobelX())
vgg_lower__covid_mini_sobel_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=1))
vgg_lower__covid_mini_sobel_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=1))

vgg_lower__covid_mini_sobel_mean_thresh = copy.deepcopy(vgg_lower__covid_mini_blog1)
vgg_lower__covid_mini_sobel_mean_thresh['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_mini_sobel_mean_thresh['traindir'] + "_" + "mean_thresh_sobel"
vgg_lower__covid_mini_sobel_mean_thresh['train_transforms'].insert(0, SobelX())
vgg_lower__covid_mini_sobel_mean_thresh['eval_transforms'].insert(0, SobelX())
vgg_lower__covid_mini_sobel_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=1))
vgg_lower__covid_mini_sobel_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=1))

vgg_lower__covid_mini_erode = copy.deepcopy(vgg_lower__covid_mini_blog1)
vgg_lower__covid_mini_erode['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_mini_erode['traindir'] + "_" + "erode"
vgg_lower__covid_mini_erode['train_transforms'].insert(0, ErosionAndDilation())
vgg_lower__covid_mini_erode['eval_transforms'].insert(0, ErosionAndDilation())

vgg_lower__covid_mini_clahe = copy.deepcopy(vgg_lower__covid_mini_blog1)
vgg_lower__covid_mini_clahe['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_mini_clahe['traindir'] + "_" + "clahe"
vgg_lower__covid_mini_clahe['train_transforms'].insert(0, ContrastThroughHistogram())
vgg_lower__covid_mini_clahe['eval_transforms'].insert(0, ContrastThroughHistogram())

vgg_lower__covid_mini_LBP = copy.deepcopy(vgg_lower__covid_mini_blog1)
vgg_lower__covid_mini_LBP['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_mini_LBP['traindir'] + "_" + "LBP"
vgg_lower__covid_mini_LBP['train_transforms'].insert(0, LocalBinaryPattern())
vgg_lower__covid_mini_LBP['eval_transforms'].insert(0, LocalBinaryPattern())

# #########################################################################################################################################################
WBC_segmented_baseline_mean = [0.6689, 0.5986, 0.7771]
WBC_segmented_baseline_std = [0.1027, 0.1482, 0.0534]

resnet_only__WBC_segmented_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_only__WBC_segmented_blog1['traindir'] = 'blood_WBC_segmented' 
resnet_only__WBC_segmented_blog1['testdir'] = 'blood_WBC_segmented'
resnet_only__WBC_segmented_blog1['labels'] = os.listdir(resnet_only__WBC_segmented_blog1['traindir'])
resnet_only__WBC_segmented_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
resnet_only__WBC_segmented_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__WBC_segmented_blog1['traindir'] + "_" + "blog1"
resnet_only__WBC_segmented_blog1['train_transforms'] = train_transforms_malaria_images_blog1
resnet_only__WBC_segmented_blog1['eval_transforms'] = eval_transforms_malaria_images_blog1
resnet_only__WBC_segmented_blog1['train_transforms'][-1] = transforms.Normalize(WBC_segmented_baseline_mean, WBC_segmented_baseline_std)
resnet_only__WBC_segmented_blog1['eval_transforms'][-1] = transforms.Normalize(WBC_segmented_baseline_mean, WBC_segmented_baseline_std)

vgg_covid19_only__WBC_segmented_blog1 = copy.deepcopy(resnet_only__WBC_segmented_blog1)
vgg_covid19_only__WBC_segmented_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(n_classes=4, learning_rate=learning_rate)'
vgg_covid19_only__WBC_segmented_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_covid19_only__WBC_segmented_blog1['traindir'] + "_" + "blog1"
vgg_covid19_only__WBC_segmented_blog1['train_transforms'].insert(len(vgg_covid19_only__WBC_segmented_blog1['train_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__WBC_segmented_blog1['eval_transforms'].insert(len(vgg_covid19_only__WBC_segmented_blog1['eval_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__WBC_segmented_blog1['train_transforms'][-1] = transforms.Normalize([0.6400], [0.1149])
vgg_covid19_only__WBC_segmented_blog1['eval_transforms'][-1] = transforms.Normalize([0.6400], [0.1149])

vgg19_pretrained_only__WBC_segmented_blog1 = copy.deepcopy(resnet_only__WBC_segmented_blog1)
vgg19_pretrained_only__WBC_segmented_blog1['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=4, learning_rate=learning_rate)'
vgg19_pretrained_only__WBC_segmented_blog1['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg19_pretrained_only__WBC_segmented_blog1['traindir'] + "_" + "blog1"

vgg16BN_pretrained_only__WBC_segmented_blog1 = copy.deepcopy(vgg19_pretrained_only__WBC_segmented_blog1)
vgg16BN_pretrained_only__WBC_segmented_blog1['model'] = 'Vgg16BNThreeChannelModelAllLayers(n_classes=4, learning_rate=learning_rate)'
vgg16BN_pretrained_only__WBC_segmented_blog1['name'] = 'Vgg16BNThreeChannelModelAllLayers' + "_" + vgg16BN_pretrained_only__WBC_segmented_blog1['traindir'] + "_" + "blog1"

resnet_lower__WBC_segmented_blog1 = copy.deepcopy(resnet_only__WBC_segmented_blog1)
resnet_lower__WBC_segmented_blog1['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
resnet_lower__WBC_segmented_blog1['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__WBC_segmented_blog1['traindir'] + "_" + "blog1"

resnet_untrained__WBC_segmented_blog1 = copy.deepcopy(resnet_only__WBC_segmented_blog1)
resnet_untrained__WBC_segmented_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate, pretrained=False)'
resnet_untrained__WBC_segmented_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_untrained__WBC_segmented_blog1['traindir'] + "_" + "blog1_untrained"

vgg_only__WBC_segmented_blog1 = copy.deepcopy(resnet_only__WBC_segmented_blog1)
vgg_only__WBC_segmented_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
vgg_only__WBC_segmented_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_only__WBC_segmented_blog1['traindir'] + "_" + "blog1"

vgg_lower__WBC_segmented_blog1 = copy.deepcopy(resnet_only__WBC_segmented_blog1)
vgg_lower__WBC_segmented_blog1['model'] = 'VggModelLowerLayers(freeze=False, n_classes=4, learning_rate=learning_rate)'
vgg_lower__WBC_segmented_blog1['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__WBC_segmented_blog1['traindir'] + "_" + "blog1"

vgg_untrained__WBC_segmented_blog1 = copy.deepcopy(vgg_only__WBC_segmented_blog1)
vgg_untrained__WBC_segmented_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate, pretrained=False)'
vgg_untrained__WBC_segmented_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_untrained__WBC_segmented_blog1['traindir'] + "_" + "blog1_untrained"

cnn_only__WBC_segmented_blog1 = copy.deepcopy(resnet_only__WBC_segmented_blog1)
cnn_only__WBC_segmented_blog1['model'] = 'CNNModel(n_classes=4, learning_rate=learning_rate)'
cnn_only__WBC_segmented_blog1['name'] = 'CNNModel' + "_" + cnn_only__WBC_segmented_blog1['traindir'] + "_" + "blog1"

cnn_mini__WBC_segmented_blog1 = copy.deepcopy(resnet_only__WBC_segmented_blog1)
cnn_mini__WBC_segmented_blog1['model'] = 'CNNMiniModel(n_classes=4, learning_rate=learning_rate)'
cnn_mini__WBC_segmented_blog1['name'] = 'CNNMiniModel' + "_" + cnn_mini__WBC_segmented_blog1['traindir'] + "_" + "blog1"

cnn_only__WBC_segmented_fullsize = copy.deepcopy(cnn_only__WBC_segmented_blog1)
cnn_only__WBC_segmented_fullsize['train_transforms'] = train_transforms_malaria_images_fullsize
cnn_only__WBC_segmented_fullsize['eval_transforms'] = eval_transforms_malaria_images_fullsize
cnn_only__WBC_segmented_fullsize['name'] = 'CNNModel' + "_" + cnn_only__WBC_segmented_fullsize['traindir'] + "_" + "fullsize"

cnn_only__WBC_segmented_tophat_otsu_thresh = copy.deepcopy(cnn_only__WBC_segmented_blog1)
cnn_only__WBC_segmented_tophat_otsu_thresh['name'] = 'CNNModel' + "_" + cnn_only__WBC_segmented_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
cnn_only__WBC_segmented_tophat_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=3))
cnn_only__WBC_segmented_tophat_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=3))
cnn_only__WBC_segmented_tophat_otsu_thresh['train_transforms'].insert(0, TopHat())
cnn_only__WBC_segmented_tophat_otsu_thresh['eval_transforms'].insert(0, TopHat())

cnn_only__WBC_segmented_tophat_mean_thresh = copy.deepcopy(cnn_only__WBC_segmented_blog1)
cnn_only__WBC_segmented_tophat_mean_thresh['name'] = 'CNNModel' + "_" + cnn_only__WBC_segmented_tophat_mean_thresh['traindir'] + "_" + "tophat_mean_thresh"
cnn_only__WBC_segmented_tophat_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=3))
cnn_only__WBC_segmented_tophat_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=3))
cnn_only__WBC_segmented_tophat_mean_thresh['train_transforms'].insert(0, TopHat())
cnn_only__WBC_segmented_tophat_mean_thresh['eval_transforms'].insert(0, TopHat())

cnn_only__WBC_segmented_sobel_otsu_thresh = copy.deepcopy(cnn_only__WBC_segmented_blog1)
cnn_only__WBC_segmented_sobel_otsu_thresh['name'] = 'CNNModel' + "_" + cnn_only__WBC_segmented_sobel_otsu_thresh['traindir'] + "_" + "otsu_thresh_sobel"
cnn_only__WBC_segmented_sobel_otsu_thresh['train_transforms'].insert(0, SobelX())
cnn_only__WBC_segmented_sobel_otsu_thresh['eval_transforms'].insert(0, SobelX())
cnn_only__WBC_segmented_sobel_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=3))
cnn_only__WBC_segmented_sobel_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=3))

cnn_only__WBC_segmented_sobel_mean_thresh = copy.deepcopy(cnn_only__WBC_segmented_blog1)
cnn_only__WBC_segmented_sobel_mean_thresh['name'] = 'CNNModel' + "_" + cnn_only__WBC_segmented_sobel_mean_thresh['traindir'] + "_" + "mean_thresh_sobel"
cnn_only__WBC_segmented_sobel_mean_thresh['train_transforms'].insert(0, SobelX())
cnn_only__WBC_segmented_sobel_mean_thresh['eval_transforms'].insert(0, SobelX())
cnn_only__WBC_segmented_sobel_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=3))
cnn_only__WBC_segmented_sobel_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=3))

cnn_only__WBC_segmented_erode = copy.deepcopy(cnn_only__WBC_segmented_blog1)
cnn_only__WBC_segmented_erode['name'] = 'CNNModel' + "_" + cnn_only__WBC_segmented_erode['traindir'] + "_" + "erode"
cnn_only__WBC_segmented_erode['train_transforms'].insert(0, ErosionAndDilation())
cnn_only__WBC_segmented_erode['eval_transforms'].insert(0, ErosionAndDilation())

cnn_only__WBC_segmented_clahe = copy.deepcopy(cnn_only__WBC_segmented_blog1)
cnn_only__WBC_segmented_clahe['name'] = 'CNNModel' + "_" + cnn_only__WBC_segmented_clahe['traindir'] + "_" + "clahe"
cnn_only__WBC_segmented_clahe['train_transforms'].insert(0, ContrastThroughHistogram())
cnn_only__WBC_segmented_clahe['eval_transforms'].insert(0, ContrastThroughHistogram())

cnn_only__WBC_segmented_LBP = copy.deepcopy(cnn_only__WBC_segmented_blog1)
cnn_only__WBC_segmented_LBP['name'] = 'CNNModel' + "_" + cnn_only__WBC_segmented_LBP['traindir'] + "_" + "LBP"
cnn_only__WBC_segmented_LBP['train_transforms'].insert(0, LocalBinaryPattern())
cnn_only__WBC_segmented_LBP['eval_transforms'].insert(0, LocalBinaryPattern())

cnn_only__WBC_segmented_blog1_venn = copy.deepcopy(cnn_only__WBC_segmented_blog1)
cnn_only__WBC_segmented_blog1_venn['name'] = 'CNNModel' + "_" + cnn_only__WBC_segmented_blog1_venn['traindir'] + "_" + "blog1_venn"
cnn_only__WBC_segmented_blog1_venn['venn_data'] = './CNNModel_blood_WBC_segmented_blog1_venn_data.results39_5.csv'

# #########################################################################################################################################################
covid_lungs_baseline_mean = [0.5942, 0.5939, 0.5937]
covid_lungs_baseline_std = [0.2939, 0.2941, 0.2940]

resnet_only__covid_lungs_blog1 = copy.deepcopy(vgg_only__kaggle_blog1)
resnet_only__covid_lungs_blog1['traindir'] = 'covid_lungs_ct' 
resnet_only__covid_lungs_blog1['testdir'] = 'covid_lungs_ct'
resnet_only__covid_lungs_blog1['labels'] = os.listdir(resnet_only__covid_lungs_blog1['traindir'])
resnet_only__covid_lungs_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=2, learning_rate=learning_rate)'
resnet_only__covid_lungs_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_only__covid_lungs_blog1['traindir'] + "_" + "blog1"
resnet_only__covid_lungs_blog1['train_transforms'] = train_transforms_malaria_images_blog1
resnet_only__covid_lungs_blog1['eval_transforms'] = eval_transforms_malaria_images_blog1
resnet_only__covid_lungs_blog1['train_transforms'][-1] = transforms.Normalize(covid_lungs_baseline_mean, covid_lungs_baseline_std)
resnet_only__covid_lungs_blog1['eval_transforms'][-1] = transforms.Normalize(covid_lungs_baseline_mean, covid_lungs_baseline_std)

vgg_covid19_only__covid_lungs_blog1 = copy.deepcopy(resnet_only__covid_lungs_blog1)
vgg_covid19_only__covid_lungs_blog1['model'] = 'Vgg19OneChannelModelAllLayersCovid19(n_classes=2, learning_rate=learning_rate)'
vgg_covid19_only__covid_lungs_blog1['name'] = 'Vgg19OneChannelModelAllLayersCovid19' + "_" + vgg_covid19_only__covid_lungs_blog1['traindir'] + "_" + "blog1"
vgg_covid19_only__covid_lungs_blog1['train_transforms'].insert(len(vgg_covid19_only__covid_lungs_blog1['train_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__covid_lungs_blog1['eval_transforms'].insert(len(vgg_covid19_only__covid_lungs_blog1['eval_transforms']) - 2, transforms.Grayscale(num_output_channels=1))
vgg_covid19_only__covid_lungs_blog1['train_transforms'][-1] = transforms.Normalize([0.5940], [0.2940])
vgg_covid19_only__covid_lungs_blog1['eval_transforms'][-1] = transforms.Normalize([0.5940], [0.2940])

vgg19_pretrained_only__covid_lungs_blog1 = copy.deepcopy(resnet_only__covid_lungs_blog1)
vgg19_pretrained_only__covid_lungs_blog1['model'] = 'Vgg19ThreeChannelModelAllLayers(n_classes=2, learning_rate=learning_rate)'
vgg19_pretrained_only__covid_lungs_blog1['name'] = 'Vgg19ThreeChannelModelAllLayers' + "_" + vgg19_pretrained_only__covid_lungs_blog1['traindir'] + "_" + "blog1"

vgg16BN_pretrained_only__covid_lungs_blog1 = copy.deepcopy(vgg19_pretrained_only__covid_lungs_blog1)
vgg16BN_pretrained_only__covid_lungs_blog1['model'] = 'Vgg16BNThreeChannelModelAllLayers(n_classes=2, learning_rate=learning_rate)'
vgg16BN_pretrained_only__covid_lungs_blog1['name'] = 'Vgg16BNThreeChannelModelAllLayers' + "_" + vgg16BN_pretrained_only__covid_lungs_blog1['traindir'] + "_" + "blog1"

resnet_lower__covid_lungs_blog1 = copy.deepcopy(resnet_only__covid_lungs_blog1)
resnet_lower__covid_lungs_blog1['model'] = 'ResNet18ModelLowerLayers(freeze=False, n_classes=2, learning_rate=learning_rate)'
resnet_lower__covid_lungs_blog1['name'] = 'ResNet18ModelLowerLayers' + "_" + resnet_lower__covid_lungs_blog1['traindir'] + "_" + "blog1"

resnet_untrained__covid_lungs_blog1 = copy.deepcopy(resnet_only__covid_lungs_blog1)
resnet_untrained__covid_lungs_blog1['model'] = 'ResNet18ModelAllLayers(freeze=False, n_classes=2, learning_rate=learning_rate, pretrained=False)'
resnet_untrained__covid_lungs_blog1['name'] = 'ResNet18ModelAllLayers' + "_" + resnet_untrained__covid_lungs_blog1['traindir'] + "_" + "blog1_untrained"

vgg_only__covid_lungs_blog1 = copy.deepcopy(resnet_only__covid_lungs_blog1)
vgg_only__covid_lungs_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=2, learning_rate=learning_rate)'
vgg_only__covid_lungs_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_only__covid_lungs_blog1['traindir'] + "_" + "blog1"

vgg_lower__covid_lungs_blog1 = copy.deepcopy(resnet_only__covid_lungs_blog1)
vgg_lower__covid_lungs_blog1['model'] = 'VggModelLowerLayers(freeze=False, n_classes=2, learning_rate=learning_rate)'
vgg_lower__covid_lungs_blog1['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_lungs_blog1['traindir'] + "_" + "blog1"

vgg_untrained__covid_lungs_blog1 = copy.deepcopy(vgg_only__covid_lungs_blog1)
vgg_untrained__covid_lungs_blog1['model'] = 'VggModelAllLayers(freeze=False, n_classes=2, learning_rate=learning_rate, pretrained=False)'
vgg_untrained__covid_lungs_blog1['name'] = 'VggModelAllLayers' + "_" + vgg_untrained__covid_lungs_blog1['traindir'] + "_" + "blog1_untrained"

cnn_only__covid_lungs_blog1 = copy.deepcopy(resnet_only__covid_lungs_blog1)
cnn_only__covid_lungs_blog1['model'] = 'CNNModel(n_classes=2, learning_rate=learning_rate)'
cnn_only__covid_lungs_blog1['name'] = 'CNNModel' + "_" + cnn_only__covid_lungs_blog1['traindir'] + "_" + "blog1"

cnn_mini__covid_lungs_blog1 = copy.deepcopy(resnet_only__covid_lungs_blog1)
cnn_mini__covid_lungs_blog1['model'] = 'CNNMiniModel(n_classes=2, learning_rate=learning_rate)'
cnn_mini__covid_lungs_blog1['name'] = 'CNNMiniModel' + "_" + cnn_mini__covid_lungs_blog1['traindir'] + "_" + "blog1"

cnn_only__covid_lungs_fullsize = copy.deepcopy(cnn_only__covid_lungs_blog1)
cnn_only__covid_lungs_fullsize['train_transforms'] = train_transforms_malaria_images_fullsize
cnn_only__covid_lungs_fullsize['eval_transforms'] = eval_transforms_malaria_images_fullsize
cnn_only__covid_lungs_fullsize['name'] = 'CNNModel' + "_" + cnn_only__covid_lungs_fullsize['traindir'] + "_" + "fullsize"

vgg_only__covid_lungs_tophat_otsu_thresh = copy.deepcopy(vgg_only__covid_lungs_blog1)
vgg_only__covid_lungs_tophat_otsu_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__covid_lungs_tophat_otsu_thresh['traindir'] + "_" + "tophat_otsu_thresh"
vgg_only__covid_lungs_tophat_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=50))
vgg_only__covid_lungs_tophat_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=50))
vgg_only__covid_lungs_tophat_otsu_thresh['train_transforms'].insert(0, TopHat())
vgg_only__covid_lungs_tophat_otsu_thresh['eval_transforms'].insert(0, TopHat())

vgg_only__covid_lungs_tophat_mean_thresh = copy.deepcopy(vgg_only__covid_lungs_blog1)
vgg_only__covid_lungs_tophat_mean_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__covid_lungs_tophat_mean_thresh['traindir'] + "_" + "tophat_mean_thresh"
vgg_only__covid_lungs_tophat_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=50))
vgg_only__covid_lungs_tophat_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=50))
vgg_only__covid_lungs_tophat_mean_thresh['train_transforms'].insert(0, TopHat())
vgg_only__covid_lungs_tophat_mean_thresh['eval_transforms'].insert(0, TopHat())

vgg_only__covid_lungs_sobel_otsu_thresh = copy.deepcopy(vgg_only__covid_lungs_blog1)
vgg_only__covid_lungs_sobel_otsu_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__covid_lungs_sobel_otsu_thresh['traindir'] + "_" + "otsu_thresh_sobel"
vgg_only__covid_lungs_sobel_otsu_thresh['train_transforms'].insert(0, SobelX())
vgg_only__covid_lungs_sobel_otsu_thresh['eval_transforms'].insert(0, SobelX())
vgg_only__covid_lungs_sobel_otsu_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=50))
vgg_only__covid_lungs_sobel_otsu_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='otsu', adaptive=False, cutoff=50))

vgg_only__covid_lungs_sobel_mean_thresh = copy.deepcopy(vgg_only__covid_lungs_blog1)
vgg_only__covid_lungs_sobel_mean_thresh['name'] = 'VggModelAllLayers' + "_" + vgg_only__covid_lungs_sobel_mean_thresh['traindir'] + "_" + "mean_thresh_sobel"
vgg_only__covid_lungs_sobel_mean_thresh['train_transforms'].insert(0, SobelX())
vgg_only__covid_lungs_sobel_mean_thresh['eval_transforms'].insert(0, SobelX())
vgg_only__covid_lungs_sobel_mean_thresh['train_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=50))
vgg_only__covid_lungs_sobel_mean_thresh['eval_transforms'].insert(0, ApplyThreshold(typeThreshold='mean', adaptive=True, cutoff=50))

vgg_only__covid_lungs_erode = copy.deepcopy(vgg_only__covid_lungs_blog1)
vgg_only__covid_lungs_erode['name'] = 'VggModelAllLayers' + "_" + vgg_only__covid_lungs_erode['traindir'] + "_" + "erode"
vgg_only__covid_lungs_erode['train_transforms'].insert(0, ErosionAndDilation())
vgg_only__covid_lungs_erode['eval_transforms'].insert(0, ErosionAndDilation())

vgg_only__covid_lungs_clahe = copy.deepcopy(vgg_only__covid_lungs_blog1)
vgg_only__covid_lungs_clahe['name'] = 'VggModelAllLayers' + "_" + vgg_only__covid_lungs_clahe['traindir'] + "_" + "clahe"
vgg_only__covid_lungs_clahe['train_transforms'].insert(0, ContrastThroughHistogram())
vgg_only__covid_lungs_clahe['eval_transforms'].insert(0, ContrastThroughHistogram())

vgg_lower__covid_lungs_LBP = copy.deepcopy(vgg_lower__covid_lungs_blog1)
vgg_lower__covid_lungs_LBP['name'] = 'VggModelLowerLayers' + "_" + vgg_lower__covid_lungs_LBP['traindir'] + "_" + "LBP"
vgg_lower__covid_lungs_LBP['train_transforms'].insert(0, LocalBinaryPattern())
vgg_lower__covid_lungs_LBP['eval_transforms'].insert(0, LocalBinaryPattern())

