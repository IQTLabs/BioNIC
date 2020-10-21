
 
import torch.nn as nn
import torch.nn.functional as F
import torch

import sys 
import numpy as np 
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torch
import torchvision 
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from six import add_metaclass
from contextlib import contextmanager
import random
import pickle
import os
import time

from augment import *
from dataset_prep import ImageFolderWithPaths, clean, make_weights_for_balanced_classes


print("Python: %s" % sys.version)
print("Pytorch: %s" % torch.__version__)

batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = eval(sys.argv[1])
image_transforms = {
    'train': transforms.Compose(model['train_transforms']),
    'test': transforms.Compose(model['eval_transforms']),
}

classes = clean(model['labels'])
print("classes: ", classes)

testdir = model['testdir']
if testdir == model['traindir'] and 'holdout' not in model.keys():
    testdir = "./HOLDOUT_" + model['traindir']
elif 'holdout' in model.keys():
    testdir = model['holdout']


def complete(model, data_weights, ii, files):
    #ii = str(int(ii) + 47)
    plt.hist(F.sigmoid(data_weights).squeeze().cpu().detach().numpy(), range=(0.0, 1.0))
    plt.savefig('results_' + model['name'] + str(ii) + '.png')

    file = open(model['name'] + "_venn_data.results" + str(ii) + ".csv", "w")
    file.write("filename,weight\n")
    data = list(F.sigmoid(data_weights).squeeze().cpu().detach().numpy())
    ctr = 0
    while ctr < len(files):
        file.write(files[ctr] + "," + str(data[ctr]) + "\n")
        ctr += 1
    file.close()
    print("results written to " + model['name'] + "_venn_data.results" + str(ii) + ".csv")

##############################################################################
# ReparamModule
##############################################################################
class PatchModules(type):
    def __call__(cls, *args, **kwargs):
        r"""Called when you call ReparamModule(...) """
        net = type.__call__(cls, *args, **kwargs)

        # collect weight (module, name) pairs
        # flatten weights
        w_modules_names = []
        for m in net.modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    w_modules_names.append((m, n))
            for n, b in m.named_buffers(recurse=False): # buffers are not parameters, but a part of the persistent state of the Module (nn.Module)
                                                        # for example, in BN we use running_mean and num_batches_tracked
                if b is not None:
                    print((
                        '{} contains buffer {}. The buffer will be treated as '
                        'a constant and assumed not to change during gradient '
                        'steps. If this assumption is violated (e.g., '
                        'BatchNorm*d\'s running_mean/var), the computation will '
                        'be incorrect.').format(m.__class__.__name__, n))

        net._weights_module_names = tuple(w_modules_names)

        # Put to correct device before we do stuff on parameters
        net = net.to(device)
        ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)
        assert len(set(w.dtype for w in ws)) == 1

        # reparam to a single flat parameter
        net._weights_numels = tuple(w.numel() for w in ws)
        net._weights_shapes = tuple(w.shape for w in ws)
        with torch.no_grad():
            flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

        # remove old parameters, assign the names as buffers
        for m, n in net._weights_module_names:
            delattr(m, n)
            m.register_buffer(n, None)

        # register the flat one
        net.register_parameter('flat_w', nn.Parameter(flat_w, requires_grad=True))

        return net


@add_metaclass(PatchModules)
class ReparamModule(nn.Module):
    def _apply(self, *args, **kwargs):
        rv = super(ReparamModule, self)._apply(*args, **kwargs)
        return rv

    def get_param(self, clone=False):
        if clone:
            return self.flat_w.detach().clone().requires_grad_(self.flat_w.requires_grad)
        return self.flat_w
    
    @contextmanager
    def unflatten_weight(self, flat_w):
        ws = (t.view(s) for (t, s) in zip(flat_w.split(self._weights_numels), self._weights_shapes))
        for (m, n), w in zip(self._weights_module_names, ws):
            setattr(m, n, w)
        yield
        for m, n in self._weights_module_names:
            setattr(m, n, None)            

    # reshapes the weights, and passes them into the standard forward for the model, along with the batch images
    def forward_with_param(self, batch_images, new_weights):
        with self.unflatten_weight(new_weights):
            return nn.Module.__call__(self, batch_images)

    def __call__(self, inp):
        return self.forward_with_param(inp, self.flat_w)

    # make load_state_dict work on both
    # singleton dicts containing a flattened weight tensor and
    # full dicts containing unflattened weight tensors...
    def load_state_dict(self, state_dict, *args, **kwargs):
        if len(state_dict) == 1 and 'flat_w' in state_dict:
            return super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
        with self.unflatten_weight(self.flat_w):
            flat_w = self.flat_w
            del self.flat_w
            super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
        self.register_parameter('flat_w', flat_w)

    def reset(self, inplace=True):
        if inplace:
            flat_w = self.flat_w
        else:
            flat_w = torch.empty_like(self.flat_w).requires_grad_()
        with torch.no_grad():
            with self.unflatten_weight(flat_w):
                weights_init(self)
        return flat_w
    

# a CNN model meant to mimic the number of layers of a full transfer learning model
class CNN(ReparamModule):
    def __init__(self, num_classes=10, neurons=2048):
        super(CNN, self).__init__()

        # batchnorm after activation: https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
        self.conv_base = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(), #nn.BatchNorm2d(num_features=32),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(), #nn.BatchNorm2d(num_features=64),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(), #nn.BatchNorm2d(num_features=128),

            nn.MaxPool2d(kernel_size=3, stride=2),     

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(), #nn.BatchNorm2d(num_features=256),

            nn.MaxPool2d(kernel_size=3, stride=2),   

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(), #nn.BatchNorm2d(num_features=512),

            nn.AvgPool2d(kernel_size=3, stride=2)
        )
                    
        

        self.fc = nn.Sequential(
            nn.Linear(neurons, 256),
            nn.ReLU(),
            nn.Dropout(), #nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(), #nn.BatchNorm1d(num_features=128),            
            nn.Linear(128, num_classes))
            #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        #print("in forward")
        #print(input.shape)
        output = self.conv_base(input)
        #print(output.shape)
        output = output.view(output.size(0), -1)
        #print(output.shape)
        output = self.fc(output)

        return output

class VGG(ReparamModule):
    def __init__(self, num_classes = 10):
        super(VGG, self).__init__()
        
        cfg =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                #layers += [nn.Conv2d(in_channels, in_channels, kernel_size = 2, stride = 2, padding = 0)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)                
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        #n_inputs = self.classifier[6].in_features
        
        self.classifier = nn.Sequential(    
            nn.Linear(25088, 64),
            nn.ReLU(True),
#             nn.Dropout(),
            nn.Linear(64, 64),
            nn.ReLU(True),
#             nn.Dropout(),
            nn.Linear(64, num_classes),
        )

    
    def forward(self, x):
        x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#net = VGG(num_classes=len(classes))
net = CNN(num_classes=len(classes))  

def weights_init(m):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.startswith('Conv') or classname == 'Linear':
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0.0)
            if getattr(m, 'weight', None) is not None:
                if classname == 'Linear':           
                    nn.init.xavier_normal_(m.weight)
                if classname.startswith('Conv'):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif 'Norm' in classname:
            if getattr(m, 'weight', None) is not None:
                m.weight.data.fill_(1)
            if getattr(m, 'bias', None) is not None:
                m.bias.data.zero_()
    m.apply(init_func)
    return(m)
    
class MyDataset():
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        data, target, filename = self.dataset[index]        
        return data, target, filename, index

    def __len__(self):
        return len(self.dataset)    

# uses Negative Log Likelihood; we distribute the loss through the weights and return updated weights?
def weighted_cross_entropy(logits, label, weight=None):
    reduction = 'none'
    ignore_index = -100
    l = F.nll_loss(F.log_softmax(logits, 1), label, None, None, ignore_index, None, reduction)
    return (l*weight).sum()/weight.size()[0]    

#optimizer_model = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)    
net.to(device)  
criterion = nn.CrossEntropyLoss()

cifar10_trainset = ImageFolderWithPaths(root=model['traindir'], transform=image_transforms['train'])
weights = make_weights_for_balanced_classes(cifar10_trainset.imgs, len(cifar10_trainset.classes))                                                                
weights = torch.DoubleTensor(weights)                                       
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
print('len of train', len(cifar10_trainset))
trainset = MyDataset(cifar10_trainset)
cifar10_trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True) #sampler=sampler)

cifar10_testset = ImageFolderWithPaths(root=testdir, transform=image_transforms['test'])
weights = make_weights_for_balanced_classes(cifar10_testset.imgs, len(cifar10_testset.classes))                                                                
weights = torch.DoubleTensor(weights)                                       
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
print('len of train', len(cifar10_testset))
cifar10_testloader = torch.utils.data.DataLoader(cifar10_testset, batch_size=batch_size, num_workers=2)#, sampler=sampler)

   
losses = []  
losses_syn = []
losses_miss = []
losses_miss2 = []
accuracy = []
#criterion_miss2 = nn.BCEWithLogitsLoss()
#criterion_miss = nn.L1Loss()
lr_synthetic = 1e-1 #learning rate of the weights
lr_model = 1e-4   # learning rate of the model

grad_val = []

n_epoch = 10
n_restarts = 100
data_weights = torch.zeros(len(cifar10_trainloader.dataset), requires_grad=True, device = device)
data_mom = torch.zeros(len(cifar10_trainloader.dataset), requires_grad=True, device = device) # momentum
foundNan = False

#data_weights, losses, losses_syn = pickle.load(open( model['name'] + '_data_weights_DB_class_norm_eval.pickle', "rb"))
#print(data_weights.shape)
#n_restarts = 47

b1 = 0.9
b2 = 0.999
eps = 1e-8 # a very small number, basically to avoid dividing by zero

# https://discuss.pytorch.org/t/how-to-track-and-add-autograd-computation-graphs-for-buffers/58080

'''
https://mlfromscratch.com/optimizers-explained/#/
Adam uses adaptive learning for the gradient, that is, it is a learning rate scheduler.
You divide the learning rate by a bigger and bigger number each time, so it will get smaller and smaller the more you learn. You are basically
dividing out the learning rate by the sum of what you've learned so far (focusing on more recent learnings through decay).

In normal SGD, all of the inputs have equal weight towards calculating the loss. Here, we are learning a reweigthing of the inputs, where we
are minimizing the weighted loss. There is a train and validation set, so we can learn what the weights are by learning input weights during 
training, and then see how that model does during validation.

Calculating the weights of the inputs:
1. Normal forward pass, with model weights and inputs used to calculate a training loss. The loss is weighted by the data_weights
2. Normal backwards pass, where you use the loss to update the model weights.
3. Another forward pass (clean?) to calculate a validation loss
4. Another backward pass (clean?) to propagate something about the data_weights
5. Another backward pass (backward on backward) to update the data_weights

'''

for ii in range(n_restarts):
    print('n_restart = ' + str(ii))

    # for each restart, create the model and reset the weights (a local function -- ask Felipe), and set it to train
    net = CNN(num_classes=len(classes)) 
    net.reset() # double sure that everything is back to square one; shouldn't need the previous line; for example BN and biases
    net.train()

    # gets the self.flat_w (weights) from the model
    weights = torch.tensor(net.get_param().cpu().detach().numpy(), requires_grad = True).to(device)   
    # initialize the momentum to be a zero for each weight
    mt = torch.zeros(weights.size()).to(device)
    # initialize the previously-learned update for each weight to be zero
    vt = torch.zeros(weights.size()).to(device) 
    
    old_data_weights = data_weights
    files = [""] * len(cifar10_trainloader.dataset)

    count = 1
    # 1. for each epoch
    for jj in range(n_epoch):
        t=time.time()

        # 2. 3., Get the train and validate batches
        for batch, batch_eval in zip(cifar10_trainloader, cifar10_trainloader):   
            print('data_weights', data_weights)    
            imgs, labels, filenames, ind = batch
            array_ind = list(ind.numpy())
            ctr = 0
            while ctr < len(array_ind):
                files[array_ind[ctr]] = filenames[ctr]
                ctr += 1
            imgs, labels = imgs.to(device), labels.to(device) 
            #print(imgs[0])
            #for i in labels:
            #    print(i)
            #print(files[0])
            #print(ind[0])
            #quit()
            
            imgs_eval, labels_eval, _, _ = batch_eval # KINGA: had to add _
            imgs_eval, labels_eval = imgs_eval.to(device), labels_eval.to(device) 
            
            ind = ind.to(device) # the indicies for each image; we used these to index into the data weights below

            # 5 (a). set the leanred data weights (epsilons) to zero 
            batch_data_weights = (torch.tensor(data_weights[ind], requires_grad=True, device = device)) # the data weigths for this batch of images
            print('batch_data_weights ', batch_data_weights)

            # NORMAL FORWARD PASS ###################################################################################################################################################
            #    used to calculate the normal training loss
            ## train with data
            with torch.enable_grad():   # enables gradient calculation -- allows you to back propagate
                                        # Enables gradient calculation inside a no_grad context. This has no effect outside of no_grad.
                print('weights ', weights)
                # 4. first forward pass
                output = net.forward_with_param(imgs, weights)
                print('train output ', output)
                # 5 (b). calculate the loss of the main predictions, using the epsilons
                # the data_weights are used to calculate the loss, whereas normally all inputs are weighted equal for loss calculation
                # Felipe does the sigmoid; the paper does the softmax
                loss = weighted_cross_entropy(output, labels, torch.sigmoid(batch_data_weights))
                print('train loss ', loss)

            # BACKWARD PASS #########################################################################################################################################################
            #    used the loss to update the weights of the model
            # puts the learning rate of the model into a tensor, which we need because we'll combine lr with gradients and weights later
            lr_tensor = torch.tensor(lr_model).to(device)
            print('lr_tensor ', lr_tensor)
            
            # 6. calculate the gradients for the model weights
            # calculatess the gradient_weights; a direction and a magnitude
            gradient_weights, = torch.autograd.grad(loss, weights, lr_tensor, create_graph=True) 
            print('gradient_weights ', gradient_weights)

            # 7. down by Adam optimizer step -- Felipe thinks this is an open question where it should be
            
            # zeroes out the gradient weights for the next epoch (otherwise they're accumulated, which looks like the model can't learn)
            net.zero_grad()                                            

            # shows the learning over time for the data_weights for each batch
            losses.append(loss.item() * imgs.size(0) / torch.sigmoid(batch_data_weights).sum())

            # SECOND FORWARD PASS ##################################################################################################################################################    
            with torch.enable_grad():
                # 8. makes predictions on the validation set
                output = net.forward_with_param(imgs_eval, weights) #I don't think this weights was updated by gradient_weights ever????????
                # 9. calculates the average loss for the validation set (note no epsilons/weighting here by inputs)
                print('valid output ', output)
                loss0 = criterion(output, labels_eval) # calculates CrossEntropyLoss for eval
                print('valid loss0 ', loss0)

            # BACKWARD PASS #########################################################################################################################################################
            
            # 10. calculates the gradients of the epsilons using the loss0 above
            # calculates the gradients from the weights and the loss; these are used to update the weights later
            gradient_of_epsilons, = torch.autograd.grad(loss0, (weights,))
            print('gradient_of_epsilons gradients of epsilon ', gradient_of_epsilons)

            # 11 (a). negates the gradient of the epsilon  gets the best data weight so far 
            # negative because you're trying to maximize the gradient alignment; want to make dw the gradient of the weights relative to the eval loss * grad of weights of the training loss; make is as big as possible
            gradient_of_epsilons = gradient_of_epsilons.neg() # gw is already weighted by lr, so simple negation
            print('gradient_of_epsilons.neg ', gradient_of_epsilons)

            # 11 (b). calculating the updated data weight; multiplying the lr_tensor times the outputs, then doing the backwards through that
            # Felipe's new changes do not use hvp_grad (gradient alignment versus optimizing for loss -- he changed so it optimizes for eval loss; changed model update -- moved #7 back up higher, aloows it to look at eval loss)
            # and just call backward; gives you the gradients to update the images weight
            print('gradient_weights ', gradient_weights)
            print('batch_data_weights ', batch_data_weights)
            print('gradient_of_epsilons ', gradient_of_epsilons)
            hvp_grad = torch.autograd.grad(
                outputs=(gradient_weights,),
                inputs=[batch_data_weights],
                grad_outputs=(gradient_of_epsilons,)
            )
            print('hvp_grad (updated data weights)', hvp_grad)
               
            # UPDATE DATA SCORE; new weight = old_weight - learning_rate * derivative
            # momentum is used to speed up the learning, when the slope is steep. It is typically started with 0.9, and called gamma.
            # the formula is the same as SGD, but we add in the momentum multiplied by the previous change to theta (data_weights).
            data_mom.data[ind] = 0.9 * data_mom.data[ind] + lr_synthetic * hvp_grad[0]
            print('data_mom.data[ind] after', data_mom.data[ind])
            # the dataweights (theta) are updated by subtracting out the [learning_rate * loss_funcion_we_are_trying_to_optimize + momentum calculated above]
            data_weights.data[ind] = data_weights.data[ind] - data_mom.data[ind]

            print('end data_weights', data_weights) 
            #files[ind] = filenames[ind] 
            # update model weights
            net.zero_grad()
            print('zero grad net on batch ')
            if count == 2:
                quit()


            # 12 ? calculating the loss with the updated data weights
            # 13 ? getting the gradients for the updated data weight, to be used in getting the final model weights

            # 14. optimizer step
            # this is what the optimizer is -- just doing simple SGD
            with torch.no_grad():
                #w = w.sub(gradient_weights).requires_grad_()

                # Adam optimizer below, old optimizer above

                # update the momentum, using the first expotential decay term (beta1)
                mt = b1 * mt + (1 - b1) * gradient_weights
                print('mt ', mt)
                # update the last updates, using the second exponential decay term (beta2)
                vt = b2 * vt + (1 - b2) * gradient_weights ** 2
                print('vt ', vt)
                # take the previous weights, and subtract the (learning_rate * momentum) / (sqrt_of_last_weights_update + eps)
                print('count ', count)
                print('(1 - b2 ** count) ', (1 - b2 ** count))
                print('torch.sqrt(vt/ (1 - b2 ** count)) ', torch.sqrt(vt/ (1 - b2 ** count)) )
                print('(mt / (1 - b1 ** count)) ', (mt / (1 - b1 ** count)))
                weights = weights - 1 / (torch.sqrt(vt/ (1 - b2 ** count)) + eps) * (mt / (1 - b1 ** count))
                print('updated model weights after adam ', weights)
                weights = weights.requires_grad_()
            count += 1

        ## normalize per class
        class_score = torch.zeros(len(classes))
        for i, s in enumerate(data_weights):
            class_score[trainset.dataset.targets[i]] += s
        class_norm = class_score/len(trainset)*1.0*len(classes)

        for i in range(len(trainset)):
            data_weights[i] += -class_norm[trainset.dataset.targets[i]]
        print(data_weights)
        print(data_weights.shape)
        complete(model, data_weights, str(ii) + '_' + str(jj), files)
        if torch.isnan(data_weights).any():
            print("FOUND NAN")
            print("NaN detected, leaving the restarts...")
            print(old_data_weights)
            quit()


#     output = net.forward_with_param(imgs_eval, w)
#     loss = criterion(output, labels_eval)

    

    loss_sum = 0
    for batch in cifar10_testloader:        
        imgs, labels, _ = batch
        imgs, labels = imgs.to(device), labels.to(device) 
        output = net.forward_with_param(imgs, w)
        loss = criterion(output, labels)
        loss_sum += loss.item()
    losses_syn.append(loss_sum)

    with open(model['name'] + '_data_weights_DB_class_norm_eval.pickle', 'wb') as f:
        pickle.dump([data_weights, losses, losses_syn, ], f)



   
    
    
    
    
    
    
    
    

