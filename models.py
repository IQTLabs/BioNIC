# #####################################################################################################################
''' 

This module provides various models that can be used for biomedical images, as well as code to run train and test calls 
on each epoch. It contains the basic train and test methods called on any model type in this codebase.

Different types of models are available:
- different transfer learning options from ImageNet (vgg-s ResNet-s)
- transfer learning from COVID19 datasets
- CNN models built from scratch
- Localized Binary Pattern based models 

'''
# #####################################################################################################################

# PyTorch
from torchvision import models
from torch import optim, cuda
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

# data science imports
import os
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt

from dataset_prep import weighted_accuracy

PRINT_ACTIVATIONS = False # see line 63 to adjust for architecture; used to print out activations for a blog post

# #####################################################################################################################
# BASE CLASS with train and test
# #####################################################################################################################

activation = {}
def get_activation(name):
    """ used with PRINT_ACTIVATIONS if we want to do so for a single image (to include in a blog post) """
    def hook(model, input, output):
        activation[name] = output 
    return hook

class Model():
    """ the base model class used in all our experiments """

    def __init__(self, name, model, learning_rate, optimizer='adam', loss_func=nn.CrossEntropyLoss()):
        """ 
        Args:
            name: a human-readable name for your model
            model: the actual model object, e.g. torchvision.models.resnet18(pretrained=True)
            learning_rate: learning rate used in your optimizer
            optimizer: a string flag to use optim.Adam, otherwise your actual optimizer object
            loss_func: whatever loss function you choose, default is cross entropy
        """
        self.name = name
        self.model = model

        if optimizer == 'adam':
            if "ResNet" in name:
                self.optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        self.loss_fn = loss_func

        # Move to gpu and parallelize
        if cuda.is_available():
            self.model = self.model.to('cuda')

    def train(self, train_loader, val_loader, epochs, device, model_options, versionNum=''):
        """ train, and optionally validate, a model 

        Args:
            train_loader: DataLoader with your training images
            val_loader: DataLoader with your validation images, or None
            epochs: number of epochs to train your model
            device: should be "cuda" or "cpu"
            model_options: a dict of model options you specified in yout *_config.py file
            versionNum: optional human-readable string to add in to your .torch filename generated here
        """

        self.model.to(device)

        #https://discuss.pytorch.org/t/visualize-feature-map/29597/2
        if PRINT_ACTIVATIONS:
            #self.model.features[15].register_forward_hook(get_activation('conv1')) #vgg_all
            self.model.conv_base[10].register_forward_hook(get_activation('conv1')) #cnn_all

        # option to use a scheduler for the learning rate
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=int(epochs/2), gamma=0.1)

        curve = open(model_options['name'] + model_options['file_label'] + ".curve", "a+")
        curve.write("epoch,training_loss,weighted_accuracy_train\n")
        for epoch in range(epochs):
            self.model.train()  # put the model in training mode

            # set up the metics we care about
            training_loss = 0.0
            valid_loss = 0.0
            num_correct = 0
            num_examples = 0
            num_correct_shape = {}
            all_targets = []
            all_preds = []

            # run each train batch through the model
            ctr = 0
            for batch in train_loader:
                if ctr % 1000 == 0:
                    print("in batch train " + str(ctr) + " of " + str(len(train_loader)))
                self.optimizer.zero_grad()

                # pass the training data through the model
                inputs, targets, _ = batch
                inputs = inputs.float()
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = self.model(inputs)

                # calculate the loss and correctness metrics
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
                predicted = torch.max(F.softmax(output, dim=1), dim=1)[1].cpu().numpy()
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted)      

                loss = self.loss_fn(output, targets)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item() * inputs.size(0)
                ctr += 1

            training_loss /= len(train_loader.dataset)

            # note: we sometimes are skipping validation, to save data/time, and because we're not fine-tuning models in 
            # our experiments
            if val_loader != None:
                num_correct = 0
                num_examples = 1
                all_targets = []
                all_preds = []
                self.model.eval()
                ctr = 0
                for batch in val_loader:
                    if ctr % 1000 == 0:
                        print("in batch eval " + str(ctr) + " of " + str(len(val_loader)))
                    inputs, targets, _ = batch
                    inputs = inputs.to(device)
                    inputs = inputs.float()
                    output = self.model(inputs)
                    targets = targets.to(device)

                    loss = self.loss_fn(output, targets)

                    valid_loss += loss.item() * inputs.size(0)
                    correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
                    num_correct += torch.sum(correct).item()
                    num_examples += correct.shape[0]
                    predicted = torch.max(F.softmax(output, dim=1), dim=1)[1].cpu().numpy()
                    all_targets.extend(targets.cpu().numpy())
                    all_preds.extend(predicted)
                    ctr += 1
                valid_loss /= len(val_loader.dataset)
                result = 'Epoch: {}, Training Loss: {:.2f},Validation Loss: {:.2f}, accuracy = {:.2f}, weighted accuracy on valid: {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples, 
                    weighted_accuracy(all_preds, all_targets))
            
            else:
                valid_loss = 0.0
                result = 'Epoch: {}, Training Loss: {:.2f},Validation Loss: {:.2f}, accuracy = {:.2f}, weighted accuracy on train: {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples, 
                    weighted_accuracy(all_preds, all_targets))
                curve.write(str(epoch)+","+str(training_loss)+","+str(weighted_accuracy(all_preds, all_targets))+"\n")

            # if we set the scheduler above, we need to update it
            #self.scheduler.step()

            print(result)
            print('finished version ' + str(versionNum))
            print("saving checkpoint ./" + model_options['name'] + "_" + str(versionNum) + ".torch")
            torch.save(self.model.state_dict(), "./" + model_options['name'] + "_" + str(versionNum) + ".torch")
            print(type(self.model))
        curve.close()

    def test(self, testloader, device, model_options, aggregateResults, test_group):
        """ test a model that has been trained, to return predictions

        Args:
            testloader: DataLoader with your test images; should also contain the paths to the images (ImageFolderWithPaths)
            device: should be "cuda" or "cpu"
            model_options: a dict of model options you specified in yout *_config.py file
            aggregateResults: filename of place to store results for each test, or None
            test_group: specifies either 'test' (unique to each CVfold) or 'holdout' (stays the same across all CVfolds) 
                in the outgoing DataFrame we create below

        Returns:
            all_preds: all the predicted labels for this test set 
            all_targets: all the target labels for this test set
            all_confidences: the confidences of each prediction
            all_paths: the file path of the image of each prediction

        """

        self.model.eval() # put the model in eval mode (so it's not going to learn)

        with torch.no_grad():

            # set up the metrics we care about
            num_correct = 0
            num_examples = 0
            num_correct_shape = {}
            all_targets = []
            all_preds = []
            all_confidences = []
            all_paths = []

            # run each test batch through the model
            for batch in  testloader:
                inputs, targets, paths = batch
                inputs = inputs.float()
                inputs = inputs.to(torch.device(device))
                output = self.model(inputs)
                confidences = torch.max(F.softmax(output, dim=1), dim=1)[0]

                # calculate our metrics
                targets = targets.to(torch.device(device))
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
                predicted = torch.max(F.softmax(output, dim=1), dim=1)[1].cpu().numpy()
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted)
                all_confidences.extend(confidences)
                all_paths.extend(paths)

                # if we want to plot the activations for our blog post
                if PRINT_ACTIVATIONS:
                    act = activation['conv1'].squeeze().cpu().detach().numpy()[0]
                    print(act.shape)
                    plt.axis('off')

                    # plot all 64 maps in an 8x8 squares
                    square = 8
                    ix = 1
                    for _ in range(square):
                        for _ in range(square):
                            # specify subplot and turn of axis
                            ax = plt.subplot(square, square, ix)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # plot filter channel in grayscale
                            plt.imshow(act[ix-1], cmap='gray')
                            ix += 1
                    # show the figure
                    plt.show()
                    plt.savefig('activation_all.png')
                    quit()

            print(confusion_matrix(all_preds, all_targets))

            # calculate and print the metrics we care about -- force the metric to be zero if the model always predicts 
            # the same thing (bad model)
            if len(set(all_preds)) != 1:
                weighted = weighted_accuracy(all_preds, all_targets)
                #print("test accuracy: " + str(num_correct * 1.0 / num_examples))
                #print('weighted accuracy: ', balanced_accuracy_score(all_preds, all_targets))
                #print('weighted precision: ', precision_score(all_preds, all_targets, average='macro'))
                #print('weighted f1: ', f1_score(all_preds, all_targets, average='macro'))
            else:
                print("Only one class present in y_true. Recording 0.0 for performance.")
                weighted = 0.0        
            print('weighted accuracy: ', weighted)

            # append the results to our logging
            if aggregateResults != None:
                file = open(aggregateResults, "a+")
                file.write(model_options['traindir'] + "," + model_options['name'] + "," + test_group + "," +  \
                    str(weighted) + "," + str(datetime.now()) + "\n")
                file.close()
                print("saved results to " + aggregateResults)

        # return the results to store in the main code for this test, so we can use that to calculate a vote later
        return all_preds, all_targets, all_confidences, all_paths

    def predict(self, input):
        """ # predict for a single image; used to print out activations for that image to include in blog posts """
        output = self.model(inputs)

    def count_parameters(self):
        """ returns the number of trainable parameters, so we can report this in our blog posts when comparing 
            models """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# #####################################################################################################################
# TRANSFER LEARNING base classes
# #####################################################################################################################

class Identity(nn.Module):
    """ an identity layer, used to "get rid of" layers in our ImageNet models that we don't want to use; it just passes 
        the data forward """
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ResNet50ModelAllLayers(Model):
    """ ResNet50 model using all layers with two LL+BN for a classifier """
    def __init__(self, freeze, n_classes, learning_rate):
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = freeze

        n_inputs = model.fc.in_features

        # Add on classifier
        #model.fc = nn.Linear(n_inputs, n_classes)
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), 
            nn.ReLU(), 
            #nn.Dropout(0.4), 
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, n_classes))

        Model.__init__(self, "ResNet50ModelAllLayers", model, learning_rate)

class ResNet18ModelAllLayers(Model):
    """ ResNet18 model using all layers with two LL+BN for a classifier """
    def __init__(self, freeze, n_classes, learning_rate, pretrained=True):
        model = models.resnet18(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = freeze

        model.fc = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(), 
            #nn.Dropout(0.4), 
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, n_classes))

        Model.__init__(self, "ResNet18ModelAllLayers", model, learning_rate)

class ResNet18ModelLowerLayers(Model):
    """ ResNet18 model using first two blocks with three LL+BN for a classifier """
    def __init__(self, freeze, n_classes, learning_rate):
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = freeze

        # Remove last two layers, and add on classifier
        model.layer3 = Identity()
        model.layer4 = Identity()
        model.fc =  nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(),        
            #nn.Dropout(0.4), 
            nn.BatchNorm1d(num_features=64),            
            nn.Linear(64, 32), 
            nn.ReLU(),    
            #nn.Dropout(0.4), 
            nn.BatchNorm1d(num_features=32),            
            nn.Linear(32, n_classes))   

        Model.__init__(self, "ResNet18ModelLowerLayers", model, learning_rate)

class VggModelAllLayers(Model):
    """ Vgg16 model using all layers with two LL+BN for a classifier """
    def __init__(self, freeze, n_classes, learning_rate, pretrained=True):
        model = models.vgg16(pretrained=pretrained)
        for param in model.features.parameters():
            param.requires_grad = freeze

        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, n_classes))
        model.classifier[6].out_features = n_classes

        Model.__init__(self, "VggModelAllLayers", model, learning_rate)

class Vgg19ModelAllLayers(Model):
    """ Vgg19 model using all layers with two LL+BN for a classifier """
    def __init__(self, freeze, n_classes, learning_rate):
        model = models.vgg19_bn(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = freeze

        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), 
            nn.ReLU(), 
            #nn.Dropout(0.4),             
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, n_classes))
        model.classifier[6].out_features = n_classes

        Model.__init__(self, "Vgg19ModelAllLayers", model, learning_rate)

class VggModelLowerLayers(Model):
    """ Vgg16 model using first 13 layers only with three LL+BN for a classifier """
    def __init__(self, freeze, n_classes, learning_rate):
        model = models.vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = freeze

        # set the upper layers of the model to Identity, so they are effectively ignored/removed
        model.features[14] = Identity()
        model.features[15] = Identity()
        model.features[16] = Identity()
        model.features[17] = Identity()
        model.features[18] = Identity()
        model.features[19] = Identity()
        model.features[20] = Identity()
        model.features[21] = Identity()
        model.features[22] = Identity()
        model.features[23] = Identity()
        model.features[24] = Identity()
        model.features[25] = Identity()
        model.features[26] = Identity()
        model.features[27] = Identity()
        model.features[28] = Identity()
        model.features[29] = Identity()
        model.features[30] = Identity()
        model.classifier[0] = Identity()
        model.classifier[1] = Identity()
        model.classifier[2] = Identity()
        model.classifier[3] = Identity()
        model.classifier[4] = Identity()
        model.classifier[5] = Identity()

        model.classifier[6] = nn.Sequential(
            nn.Linear(12544, 4096), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features=4096),
            #nn.Dropout(0.4),
            nn.Linear(4096, 256), 
            nn.ReLU(), 
            nn.BatchNorm1d(num_features=256),
            #nn.Dropout(0.4),            
            nn.Linear(256, n_classes))
        model.classifier[6].out_features = n_classes

        Model.__init__(self, "VggModelLowerLayers", model, learning_rate)

# #####################################################################################################################
# BESPOKE model classes
# #####################################################################################################################

class CNN(nn.Module):
    """ a CNN model meant to mimic the number of layers of a full transfer learning model """
    def __init__(self, num_classes=10, neurons=2048):
        super(CNN, self).__init__()

        # batchnorm after activation: https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
        self.conv_base = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.MaxPool2d(kernel_size=3, stride=2),     

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),

            nn.MaxPool2d(kernel_size=3, stride=2),   

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),

            nn.AvgPool2d(kernel_size=3, stride=2)
        )
                    
        self.fc = nn.Sequential(
            nn.Linear(neurons, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),            
            nn.Linear(128, num_classes))

    def forward(self, input):
        output = self.conv_base(input)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

class CNNGreyscale(nn.Module):
    """ a CNN model with only a single incoming channel """
    def __init__(self, num_classes=10):
        super(CNNGreyscale, self).__init__()

        # batchnorm after activation: https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
        self.conv_base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=10),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=20),

            #nn.MaxPool2d(kernel_size=3, stride=2),

            #nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(num_features=64),

            #nn.MaxPool2d(kernel_size=3, stride=2),     

            #nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(num_features=256),

            #nn.MaxPool2d(kernel_size=3, stride=2), 

            #nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm2d(num_features=512),

            nn.AvgPool2d(kernel_size=4, stride=2)
        )
                    
        self.fc = nn.Sequential(
            nn.Linear(162000, 1024),
            nn.ReLU(),
            nn.Dropout(0.5), 
            #nn.BatchNorm1d(num_features=256),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            #nn.BatchNorm1d(num_features=128),            
            nn.Linear(128, num_classes))

    def forward(self, input):
        output = self.conv_base(input)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

class CNNMini(nn.Module):
    """ a CNN model with fewer layers than the full one defined above """
    def __init__(self, num_classes=10):
        super(CNNMini, self).__init__()

        # batchnorm after activation: https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
        self.conv_base = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=10),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=20),

            nn.AvgPool2d(kernel_size=3, stride=2) 
        )

        self.fc = nn.Sequential(
            nn.Linear(54080, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),            
            nn.Linear(128, num_classes))

    def forward(self, input):
        output = self.conv_base(input)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

# #####################################################################################################################
# WRAPPER classes that call the custom models we created above
# #####################################################################################################################

class CNNMiniModel(Model):
    def __init__(self, n_classes, learning_rate):
        model = CNNMini(n_classes)
        Model.__init__(self, "CNNMiniModel", model, learning_rate)

class CNNModel(Model):
    def __init__(self, n_classes, learning_rate, neurons=2048):
        model = CNN(n_classes, neurons)
        Model.__init__(self, "CNNModel", model, learning_rate)

class CNNGreyModel(Model):
    def __init__(self, n_classes, learning_rate):
        model = CNNGreyscale(n_classes)
        Model.__init__(self, "CNNGreyModel", model, learning_rate)

class CellNetBiomedShapeModel(Model):
    """ Creates a bespoke greyscale CNN that was pre-trained on the biomed_self_label_shape dataset; this is meant to 
        be used for transfer learning """
    def __init__(self, freeze, n_classes, learning_rate):
        model = CNNGreyscale(4) 
        model.load_state_dict(torch.load("./CNNGreyModel_biomed_self_label_shape_baseline_0.torch"))
        model.fc[6] = nn.Linear(128, n_classes)
        Model.__init__(self, "CellNetBiomedShapeModel", model, learning_rate)

class Vgg16BiomedShapeModel(Model):
    """ Allows transfer learning from a Vgg16 model tha was trained on the biomed_self_label_shape dataset """
    def __init__(self, freeze, n_classes, learning_rate):
        model = VggModelAllLayers(freeze=False, n_classes=4, learning_rate=learning_rate)
        model.model.load_state_dict(torch.load("./VggModelAllLayers_biomed_self_label_shape_baseline_0.torch"))
        model.model.classifier[6][3] = nn.Linear(256, n_classes)
        Model.__init__(self, "Vgg16BiomedShapeModel", model.model, learning_rate)

class ResNetCovid19Model3Classes(Model):
    """ Allows transfer learning from a Vgg16 model tha was trained on the biomed_self_label_shape dataset """
    def __init__(self, n_classes, learning_rate):
        model = models.resnet50(pretrained=False) 
        model.conv1 = torch.nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), 
                                      padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True) 

        chpt = torch.load('./michael/chpt_5.pth.tar')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in chpt['model'].items():
            name = k[7:] # removes 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), 
                                      padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True) 
        Model.__init__(self, "ResNetCovid19Model3Classes", model, learning_rate)


class Vgg19OneChannelModelAllLayers(Model):
    """ Classifier meant to be used with the huge covid19 images dataset; this dataset contains images with five 
        channels, one for each different treatment of a cell culture with some chemical/drug. We build a classifier 
        below to take a single-channel image, and predict which of the five channels it belongs to -- this is kind 
        of biologically meaningless, but our hope is that this model learns something that is useful for transfer 
        learning for other biomedical models. See cellnet_transfer.py for further details.

    """
    def __init__(self, n_classes, learning_rate, pretrained=False):
        model = models.vgg19_bn(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = True

        model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 32),
            nn.ReLU(), 
            nn.BatchNorm1d(num_features=32),
            #nn.Dropout(0.4),
            nn.Linear(32, n_classes))
        model.classifier[6].out_features = n_classes

        Model.__init__(self, "Vgg19OneChannelModelAllLayers", model, learning_rate)
 
class Vgg19OneChannelModelAllLayersCovid19(Model):
    """ Allows transfer learning from a Vgg19_bn model tha was trained on the covid19 greyscale dataset """
    def __init__(self, n_classes, learning_rate, saved_model):
        model = Vgg19OneChannelModelAllLayers(n_classes=5, learning_rate=learning_rate)
        model.model.load_state_dict(torch.load(saved_model, map_location=torch.device('cuda')))
        model.model.classifier[6][3] = nn.Linear(32, n_classes)

        Model.__init__(self, "Vgg19OneChannelModelAllLayersCovid19", model.model, learning_rate)

class Vgg19ThreeChannelModelAllLayers(Model):
    """ same as Vgg19OneChannelModelAllLayers, but for three channels """
    def __init__(self, n_classes, learning_rate, pretrained=True):
        print("classes ", n_classes)
        model = models.vgg19_bn(pretrained)
        for param in model.parameters():
            param.requires_grad = True

        model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 32),
            nn.ReLU(), 
            nn.BatchNorm1d(num_features=32),
            #nn.Dropout(0.4),
            nn.Linear(32, n_classes))
        
        model.classifier[6].out_features = n_classes

        Model.__init__(self, "Vgg19ThreeChannelModelAllLayers", model, learning_rate)

class Vgg19ThreeChannelModelAllLayersCovid19(Model):
    """ Allows transfer learning from a Vgg19_bn model tha was trained on the covid19 RGB dataset """
    def __init__(self, n_classes, learning_rate, saved_model):
        model = Vgg19ThreeChannelModelAllLayers(n_classes=5, learning_rate=learning_rate)
        model.model.load_state_dict(torch.load(saved_model, map_location=torch.device('cuda')))
        model.model.classifier[6][3] = nn.Linear(32, n_classes)

        Model.__init__(self, "Vgg19ThreeChannelModelAllLayersCovid19", model.model, learning_rate)

class CNNGreyModelCovid19(Model):
    """ Allows transfer learning from a shallow CNN model tha was trained on the covid19 greyscale dataset """
    def __init__(self, n_classes, learning_rate, saved_model):
        model = CNNGreyModel(n_classes=5, learning_rate=learning_rate)
        model.model.load_state_dict(torch.load(saved_model, map_location=torch.device('cuda')))
        model.model.fc[6] = nn.Linear(128, n_classes)

        Model.__init__(self, "CNNGreyModelCovid19", model.model, learning_rate)
