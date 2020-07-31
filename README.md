# BioNIC
optimizing deep learning for ultra-small biomedical datasets 

## Project description
This codebase sets up an infrastructure to train and evaluate deep learning models on ultra-small (less than 1000 samples) biomedical imagery datasets. It is meant to be used to triage biomedical images for human inspection, based on the confidence in a prediction. That is, high-confidence predictions by the models can be used directly, while lower-confidence predictions will need to be manually evaluated.

The user is responsible for dividing their images into train, evaluate, and holdout sets, and specifying the configuration parameters of the models. Multiple models can be built off the same training data, and evaluated against one another.

Some of the configuration options include:
- Choosing transfer learning from other pre-trained models
- Choosing data augmentations and transformations
- Choosing the number of CrossValidation-folds and repeats

Each model that is trained can be run on test data, where images are classified, and then these classifications are output into low and high confidence folders, indicating which ones need human inspection.

## Requirements
We use PyTorch, Pillow, and OpenCV libraries for image processing. You can view all libraries and/or install them with conda with:
`conda env create -f bionic.yaml` 

We recommend running this code with a GPU-enabled machine; it takes on the order of 30 minutes to train 10-20 CV-folds of a single model with a GPU.

## Image folder preparation
Images you plan to use should be split into train, test, and holdout folders. 

Each folder should have a sub-folder for each of the target classes (because we will use `torchvision.datasets.ImageFolder` to load data into batches). Your images can be of any size, but you'll most likely be resizing them, especially if you plan to use transfer learning from ImageNet models (that requires 224x224 sized images).

We recommend that you split you images into train and test sets with caution, especially if you are tiling your raw images, and/or you are taking multiple images from the same specimen slide, that you ensure that tiles that stem from a single source do not get split up between train/test.

## Usage instructions
The following options are available to run model or models, and view results:
- train and test a single model from the command line, using `cellnet_driver_no_validation.py`
- view the results of any previously-run model(s) using `display_results.iypnb`
- run any saved model (all models are saved by default) on a new dataset of your choosing, using `run_saved_model.py`. This will run your test data on the N verions of the previously-trained model, and provide the results of a voting ensemble model.
- train and test multiple models at once, using `driver.py`. It will also dump test images into high and low confidence folders, with each filename containing the predicted label and target label.

Models are specified in the configuration files imported; you simply need to modify or add a configuration to run a new model. Otherwise, see the individal files listed above for further instructions and settings.

### Simple run
To train your first model from the existing dataset, go to the terminal and type:

`time python3 cellnet_driver_no_validation.py 0 vgg_only__kaggle_blog1 False`

This will re-run the listed model on GPU 0. `False` tells the script not to generate a holdout test set (this model provides one manually): see `cellnet_driver_no_validation.py` for more details.

Once you have that trained model, you can test it on a holdout test set you specify in `augment_config.py` by running:

`time python3 run_saved_model.py 0 vgg_only__kaggle_blog1` 

on GPU 0.

Finally, you can view the results graphically by specifying the csv of results created above, `dataframe_VggModelAllLayers_kaggle_bowl.csv`, at the top of the second cell in `display_results.iypnb`, which you can run by opening a Jupyter notebook by typing `jupter notebook` in the root directory.

### Descriptions of included files:
- `bionic.yaml`: the list of packages you'll need to run this code; you can use conda to install it (see above).
- `cellnet_config.py`, `augment_config.py`: various config files for different models/experiments we ran. You can add your own models to any of these, or create and import your own config files. 
- `dataset_prep.py`: basic utilities to do stratified CV splits and/or global holdouts for a dataset where you want to use cross-validation to build and test a model. Also specifies the reporting metric(s), which all attempt to correct for any class imbalances during dataset augmentation and reporting.
- `augmentations.py`: various dataset transformations that can be used during your dataset preparation; see the file for more details.
- `models.py`: implements various neural net architectures to experiment with, including transfer learning models, large and small CNNs, and models that use Gabor filters. Contains the base train and test methods for all the models. 
- `cellnet_driver_no_validation.py`: train and test a single model. You can specify the number of CV-folds/iterations here, as well as the number of epochs, the device (cpu/gpu), whether or not you want to generate a global holdout (or use an exisitng one), and if you want to calculate the mean and stddev of pixels in all images in a dataset to use for normalization in your model configurations.
- `run_saved_model.py`: run any saved model (all models are saved by default) on a new dataset of your choosing. Prints filenames, predicted labels, and confidences to a csv.
- `driver.py`: train and test multiple models at once, and report their results in terms of low and high confidence model predictions.
- `display_results.iypnb`: view the results of any previously-run model(s)
- `utilities.py`: various utlities that you can use to help prep your dataset folders.
- `cellnet_pull.py`: Converts a five-channel, 100K covid19 dataset into a single-channel (greyscale) covid19 dataset with five classes; we use this to evaluate transfer learnining from a classifier built off this model.

## Further reading
We will be updating this section with results and blog posts for our experimentation using transfer learning, data transformations and augmentation, and other metrics.

