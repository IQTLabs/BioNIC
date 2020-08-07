'''

This file will run one or more models specified in any of tbe config files, save the results, and place the images by 
confidence in low and high directories.

'''
import os
import sys
from subprocess import Popen, PIPE, STDOUT

gpu = "0"
confidence_threshold = 0.8

""" List of which model(s) do you want to run? The models have all been specified in your *_config.py files
	The True flag will re-create the holdout, while False will not -- we recommend that you only set it to True for 
	the first model in a series of experiments where you want to compare different models on the same holdout dataset.

	For readability, the groupings below are for each individual dataset that you'd want to try to build and evaluate 
	different models for.
"""
models_to_run = [ 	
					#"biomed_transfer_CNN True",

					#"biomed_self_label_shape_blog1_resnetfull True",
					#"biomed_self_label_shape_blog1_resnetlower False",
					#"biomed_self_label_shape_blog1_vggfull False",
					#"biomed_self_label_shape_blog1_vgglower False",
					#"biomed_self_label_shape_blog1_cnnfull False",
					#"biomed_self_label_shape_blog1_cnnmini False",
					#"biomed_self_label_shape_blog1_cnngrey False"
					#"vgg_only__biomed_self_label_shape_centercrop False",
					##"vgg_only__biomed_self_label_shape_randomcrop False",
					#"vgg_only__biomed_self_label_shape_full_resize False",
					#"vgg_only__biomed_self_label_shape_half_resize False",
					#"vgg_only__biomed_self_label_shape_tophat_otsu_thresh False",
					#"vgg_only__biomed_self_label_shape_tophat_mean_adaptive_thresh False",
					#"vgg_only__biomed_self_label_shape_sobel_otsu_thresh False",
					#"vgg_only__biomed_self_label_shape_sobel_mean_adaptive_thresh False",
					#"vgg_only__biomed_self_label_shape_erode False",
					#"vgg_only__biomed_self_label_shape_LBP False"

					#"vgg_only__kaggle_blog1 False",
					#"vgg_lower__kaggle_blog1 False",
					#"resnet_only__kaggle_blog1 False",
					#"resnet_lower__kaggle_blog1 False",
					#"cnn_only__kaggle_blog1 False",
					#"cnn_mini__kaggle_blog1 False",
					#"vgg_only__kaggle_center_crop False",
					#"vgg_only__kaggle_center_full_resize False",
					#"vgg_only__kaggle_center_half_resize False",
					#"vgg_only__kaggle_random_crop False"
					#"resnet_only__kaggle_tophat_otsu_thresh False",
					#"resnet_only__kaggle_tophat_mean_thresh False",
					#"resnet_only__kaggle_sobel_otsu_thresh False",
					#"resnet_only__kaggle_sobel_mean_thresh False",
					#"vgg_covid19_only__kaggle_blog1 False",
					#"resnet_only__kaggle_erosion False",
					#"vgg_only__kaggle_LBP False",
					#"resnet_only__kaggle_blog1_20e False",
					#"resnet_only__kaggle_clahe False",
					#"vgg_untrained__kaggle_blog1 False",
					#"resnet_untrained__kaggle_blog1 False"
					#"vgg_covid19_only__kaggle_blog1 False",
					#"vgg19_pretrained_only__kaggle_blog1 False",
					#"vgg16BN_pretrained_only__kaggle_blog1 False",
					#"vgg_1channel_covid19_noImageNet__kaggle_blog3 False",
					#"vgg_1channel_covid19_withImageNet__kaggle_blog3 False",
					#"vgg_1channel_withImageNet_only__kaggle_blog3 False",
					#"vgg_1channel_noImageNet_only__kaggle_blog3 False",
					#"vgg_3channel_covid19_noImageNet__kaggle_blog3 False",
					#"vgg_3channel_covid19_withImageNet__kaggle_blog3 False",
					#"vgg_3channel_withImageNet_only__kaggle_blog3 False",
					#"vgg_3channel_noImageNet_only__kaggle_blog3 False",
					"cnn_grey_covid19_only__kaggle_blog3 False" 

					#"resnet_only__malaria_cell_images_blog1 False",
					#"resnet_lower__malaria_cell_images_blog1 False",
					#"vgg_only__malaria_cell_images_blog1 False",
					#"vgg_lower__malaria_cell_images_blog1 False",
					#"cnn_only__malaria_cell_images_blog1 False",
					#"cnn_mini__malaria_cell_images_blog1 False",
					#"cnn_only__malaria_cell_images_tophat_otsu_thresh False",
					#"cnn_only__malaria_cell_images_sobel_otsu_thresh False",
					#"cnn_only__malaria_cell_images_sobel_mean_thresh False",
					#"cnn_only__malaria_cell_images_tophat_mean_thresh False",
					#"cnn_only__malaria_cell_images_erode False",
					#"cnn_only__malaria_cell_images_LBP False"
					#"resnet_only__malaria_cell_images_blog1_20e False"
					#"cnn_only__malaria_cell_images_clahe False",
					#"vgg_only__malaria_cell_images_blog1 False",
					#"resnet_untrained__malaria_cell_images_blog1 False"
					#"cnn_only__malaria_cell_images_venn_blog1 False"
					#"vgg_covid19_only__malaria_cell_images_blog1 False",
					#"vgg19_pretrained_only__malaria_cell_images_blog1 False",
					#"vgg16BN_pretrained_only__malaria_cell_images_blog1 False"
			
					#"resnet_only__WBC_images_blog1 False",
					#"resnet_lower__WBC_images_blog1 False",
					#"vgg_only__WBC_images_blog1 False",
					#"vgg_lower__WBC_images_blog1 False",
					#"cnn_only__WBC_images_blog1 False",
					#"cnn_mini__WBC_images_blog1 False",
					#"cnn_only__WBC_images_tophat_otsu_thresh False",
					#"cnn_only__WBC_images_tophat_mean_thresh False",
					#"cnn_only__WBC_images_sobel_otsu_thresh False",
					#"cnn_only__WBC_images_sobel_mean_thresh False",
					#"cnn_only__WBC_images_erode False",
					#"cnn_only__WBC_images_LBP False",
					#"cnn_only__WBC_images_clahe False",
					#"vgg_untrained__WBC_images_blog1 False"
					#"resnet_untrained__WBC_images_blog1 False"
					#"vgg_covid19_only__WBC_images_blog1 False",
					#"vgg19_pretrained_only__WBC_images_blog1 False",
					#"vgg16BN_pretrained_only__WBC_images_blog1 False"

					#"resnet_only__cat_counts_blog1 False",
					#"resnet_lower__cat_counts_blog1 False",
					#"vgg_only__cat_counts_blog1 False",
					#"vgg_lower__cat_counts_blog1 False",
					#"cnn_only__cat_counts_blog1 False",
					#"cnn_mini__cat_counts_blog1 False",
					#"cnn_only__cat_counts_fullsize False",
					#"vgg_only__cat_counts_tophat_otsu_thresh False",
					#"vgg_only__cat_counts_tophat_mean_thresh False",
					#"vgg_only__cat_counts_sobel_otsu_thresh False",
					#"vgg_only__cat_counts_sobel_mean_thresh False",
					#"vgg_only__cat_counts_erode False",
					#"vgg_only__cat_counts_LBP False",
					#"vgg_only__cat_counts_clahe False",
					#"vgg_untrained__cat_counts_blog1 False",
					#"resnet_untrained__cat_counts_blog1 False"
					#"vgg_covid19_only__cat_counts_blog1 False",
					#"vgg19_pretrained_only__cat_counts_blog1 False",
					#"vgg16BN_pretrained_only__cat_counts_blog1 False"

					#"resnet_only__labeled_confocal_protein_blog1 False",
					#"resnet_lower__labeled_confocal_protein_blog1 False",
					#"vgg_only__labeled_confocal_protein_blog1 False",
					#"vgg_lower__labeled_confocal_protein_blog1 False",
					#"cnn_only__labeled_confocal_protein_blog1 False",
					#"cnn_mini__labeled_confocal_protein_blog1 False",
					#"cnn_only__labeled_confocal_protein_halfsize False",
					#"resnet_lower__labeled_confocal_protein_tophat_otsu_thresh False",
					#"resnet_lower__labeled_confocal_protein_tophat_mean_thresh False",
					#"resnet_lower__labeled_confocal_protein_sobel_otsu_thresh False",
					#"resnet_lower__labeled_confocal_protein_sobel_mean_thresh False",
					#"resnet_lower__labeled_confocal_protein_erode False", 
					#"cnn_only__labeled_confocal_protein_LBP False",
					#"resnet_only__labeled_confocal_protein_blog1_20e False"
					#"resnet_lower__labeled_confocal_protein_clahe False",
					#"cnn_only__labeled_confocal_protein__blog1_venn False",
					#"resnet_untrained__labeled_confocal_protein_blog1 False",
					#"vgg_untrained__labeled_confocal_protein_blog1 False"
					#"vgg_covid19_only__labeled_confocal_protein_blog1 False",
					#"vgg19_pretrained_only__labeled_confocal_protein_blog1 False",
					#"vgg16BN_pretrained_only__labeled_confocal_protein_blog1 False"

					#"resnet_only__CHO_images_blog1 False",
					#"resnet_lower__CHO_images_blog1 False",
					#"vgg_only__CHO_images_blog1 False",
					#"vgg_lower__CHO_images_blog1 False",
					#"cnn_only__CHO_images_blog1 False",
					#"cnn_mini__CHO_images_blog1 False",
					#"vgg_only__CHO_images_tophat_otsu_thresh False",
					#"vgg_only__CHO_images_tophat_mean_thresh False",
					#"vgg_only__CHO_images_sobel_otsu_thresh False",
					#"vgg_only__CHO_images_sobel_mean_thresh False",
					#"vgg_only__CHO_images_erode False",
					#"cnn_only__CHO_images_LBP False",
					#"resnet_only__CHO_images_blog1_20e False",
					#"vgg_only__CHO_images_clahe False",
					#"cnn_only__CHO_images__blog1_venn False"
					#"vgg_untrained__CHO_images_blog1 False"
					#"resnet_untrained__CHO_images_blog1 False"
					#"vgg_covid19_only__CHO_images_blog1 False",
					#"vgg19_pretrained_only__CHO_images_blog1 False",
					#"vgg16BN_pretrained_only__CHO_images_blog1 False"

					#"resnet_only__synimages_blog1 False",
					#"resnet_lower__synimages_blog1 False",
					#"vgg_only__synimages_blog1 False",
					#"vgg_lower__synimages_blog1 False",
					#"cnn_only__synimages_blog1 False",
					#"cnn_mini__synimages_blog1 False",
					#"vgg_lower__synimages_centercrop False",
					#"vgg_lower__synimages_tophat_otsu_thresh False",
					#"vgg_lower__synimages_tophat_mean_thresh False",
					#"vgg_lower__synimages_sobel_otsu_thresh False",
					#"vgg_lower__synimages_sobel_mean_thresh False",
					#"vgg_lower__synimages_erode False",
					#"cnn_only__synimages_LBP False",
					#"vgg_lower__synimages_clahe False",
					#"resnet_untrained__synimages_blog1 False"
					#"vgg_untrained__synimages_blog1 False"
					#"vgg_covid19_only__synimages_blog1 False",
					#"vgg19_pretrained_only__synimages_blog1 False",
					#"vgg16BN_pretrained_only__synimages_blog1 False"

					#"resnet_only__covid_mini_blog1 False",
					#"resnet_lower__covid_mini_blog1 False",
					#"vgg_only__covid_mini_blog1 False",
					#"vgg_lower__covid_mini_blog1 False",
					#"cnn_only__covid_mini_blog1 False",
					#"cnn_mini__covid_mini_blog1 False",
					#"vgg_lower__covid_mini_centercrop False",
					#"vgg_lower__covid_mini_tophat_otsu_thresh False",
					#"vgg_lower__covid_mini_tophat_mean_thresh False",
					#"vgg_lower__covid_mini_sobel_otsu_thresh False",
					#"vgg_lower__covid_mini_sobel_mean_thresh False",
					#"vgg_lower__covid_mini_erode False",
					#"vgg_lower__covid_mini_LBP False",
					#"vgg_lower__covid_mini_clahe False",
					#"vgg_untrained__covid_mini_blog1 False"
					#"resnet_untrained__covid_mini_blog1 False"

					#"resnet_only__WBC_segmented_blog1 False",
					#"resnet_lower__WBC_segmented_blog1 False",
					#"vgg_only__WBC_segmented_blog1 False",
					#"vgg_lower__WBC_segmented_blog1 False",
					#"cnn_only__WBC_segmented_blog1 False",
					#"cnn_mini__WBC_segmented_blog1 False",
					#"cnn_only__WBC_segmented_tophat_otsu_thresh False",
					#"cnn_only__WBC_segmented_tophat_mean_thresh False",
					#"cnn_only__WBC_segmented_sobel_otsu_thresh False",
					#"cnn_only__WBC_segmented_sobel_mean_thresh False",
					#"cnn_only__WBC_segmented_erode False",
					#"cnn_only__WBC_segmented_clahe False",
					#"cnn_only__WBC_segmented_LBP False",
					#"cnn_only__WBC_segmented_blog1_venn False"
					#"resnet_untrained__WBC_segmented_blog1 False"
					#"vgg_untrained__WBC_segmented_blog1 False"
					#"vgg_covid19_only__WBC_segmented_blog1 False",
					#"vgg19_pretrained_only__WBC_segmented_blog1 False",
					#"vgg16BN_pretrained_only__WBC_segmented_blog1 False",

					#"resnet_only__covid_lungs_blog1 False",
					#"resnet_lower__covid_lungs_blog1 False",
					#"vgg_only__covid_lungs_blog1 False",
					#"vgg_lower__covid_lungs_blog1 False",
					#"cnn_only__covid_lungs_blog1 False",
					#"cnn_mini__covid_lungs_blog1 False",
					#"vgg_only__covid_lungs_tophat_otsu_thresh False",
					#"vgg_only__covid_lungs_tophat_mean_thresh False",
					#"vgg_only__covid_lungs_sobel_otsu_thresh False",
					#"vgg_only__covid_lungs_sobel_mean_thresh False",
					#"vgg_only__covid_lungs_erode False",
					#"vgg_lower__covid_lungs_LBP False",
					#"vgg_only__covid_lungs_clahe False",
					#"resnet_untrained__covid_lungs_blog1 False"
					#"vgg_untrained__covid_lungs_blog1 False"
					#"vgg_covid19_only__covid_lungs_blog1 False",
					#"vgg19_pretrained_only__covid_lungs_blog1 False",
					#"vgg16BN_pretrained_only__covid_lungs_blog1 False"
	]


# call the driver for individual models
for model in models_to_run:
	os.system("python3 cellnet_driver_no_validation.py " + gpu + " " + model )

# grab the results, copy images into high and low confidence folders
p = Popen("ls *.holdout_confidences.txt", shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
returned_output = p.stdout.read()
returned_output = str(returned_output).split("\\n")
files = []
for line in returned_output:
	line = str(line)
	if line.startswith("b'"):
		line = line[2:]
	if len(line) < 4:
		continue
	files.append(line)

# create high and low confidence folders
for file in files:
	handle = file.split(".")[0]
	if os.path.exists(handle):
		os.system("rm -r " + handle)
	os.system("mkdir " + handle)
	os.system("mkdir " + handle + "/high_confidence")
	os.system("mkdir " + handle + "/low_confidence")

	file = open(file)
	data = file.readlines()
	file.close()

	# move images into high and low confidence folders, where labels have been changed to make inspection easier (did 
	# we predict correctly or not?)
	for d in data:
		if len(d) == 5:
			d = d.split(',')
			filename = d[0][2:-1]
			confidence = float(d[1][8:])
			pred = d[4].rstrip()[1:]
			target = d[5].rstrip()[1:-1]
			if confidence >= confidence_threshold:
				os.system("cp " + filename + " ./" + handle + "/high_confidence/" + filename.split("/")[-1].split(".")[0] + \
					"_target" + str(target) + "_pred" + str(pred) + "." + filename.split("/")[-1].split(".")[1])
			else:
				os.system("cp " + filename + " ./" + handle + "/low_confidence/" + filename.split("/")[-1].split(".")[0] + \
					"_target" + str(target) + "_pred" + str(pred) + "." + filename.split("/")[-1].split(".")[1])

