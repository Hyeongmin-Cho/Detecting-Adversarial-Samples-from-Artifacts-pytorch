import torch.nn.init as init
import torch.nn as nn
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
import torch.optim as optim
import torchvision.transforms as transforms
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from train_vgg19 import vgg19

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from tqdm import tqdm
import warnings

from advertorch.attacks import PGDAttack, GradientSignAttack, LinfBasicIterativeAttack, \
                                    CarliniWagnerL2Attack, DDNL2Attack, SinglePixelAttack

import models
warnings.filterwarnings(action='ignore')

class CW:
    def __init__(self, model, lr=0.005, step=1000, radius=0.3, margin=0.0, class_num=10, device = 'cuda'):
        self.model = model
        self.lr = lr
        self.step = step
        self.radius = radius
        self.margin = margin
        self.class_num = class_num
        self.device = device
        
    def get_next_label(self, label, num_class):
        return (label+1)%num_class
    
    def perturb(self,x,y):
        x = x.to(self.device)
        x_var = torch.autograd.Variable(x.clone().cuda(), requires_grad=True)
        true_label = y.cpu()
        optimizer = optim.Adam([x_var], lr=self.lr)
        target_label = self.get_next_label(label=true_label, num_class=self.class_num)
        for s in (range(self.step)):
            optimizer.zero_grad()
            total_loss = 0
            output_ori = self.model(x_var)
#             print(output_ori.shape)
            _, top2_1 = output_ori.data.cpu().topk(2)
            
            argmax11 = top2_1[:,0]
            argmax11[argmax11 == target_label] == top2_1[argmax11 == target_label,1]
#             argmax11 = argmax11.unsqueeze(1).reshape(len(argmax11),1)
            
            idx = torch.arange(0, len(argmax11))
#             idx = idx.unsqueeze(1).reshape(len(idx),1)
            
#             argmax11 = torch.cat((idx,argmax11),dim=1)
#             print(argmax11)
            
#             print(output_ori[idx,argmax11].shape)
#             print(output_ori[idx,target_label].shape)
            loss = (output_ori[idx,argmax11] - output_ori[idx,target_label] + self.margin).clamp(min=0).sum()
#             print(loss1)
#             break
            loss.backward()
            optimizer.step()
            x_var.data = torch.clamp(x_var - x, min=-self.radius, max=self.radius) + x
        return x_var

class adversarial_generator:
    def __init__(self, model_path, dataset='mnist', batch_size=100, device='cpu'):
        self.model_path = model_path
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_path = model_path
        self.device = device
        
    def data_loader(self):
        train_data, test_data = get_data(self.dataset)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, num_workers=16)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False ,num_workers=16)
        
        return train_loader, test_loader
    
    def pretrained_model(self, path):
        if self.dataset == 'mnist':
            # modelA
            print('Model loading : mnist modelA')
            classifier = models.modelA().to(self.device)
            checkpoint = torch.load(path)
            classifier.load_state_dict(checkpoint)
            

            
        elif self.dataset == 'cifar10':
            # Resnet56
#             print('Model loading : cifar10 resnet56')
#             classifier = models.resnet56().to(self.device)
#             checkpoint = torch.load(path)
#             classifier.load_state_dict(checkpoint['model_state_dict'])
            
#           VGG19
            print('Model loading : cifar10 vgg19')
            classifier = vgg19()
#             classifier.features = torch.nn.DataParallel(classifier.features)
            classifier = classifier.to(self.device)
            checkpoint = torch.load(path)
            classifier.load_state_dict(checkpoint)
            
        else:
            assert False, 'There\'s no such dataset'

        classifier.eval()
        
        return classifier
    
    def classifier_adv(self, data):
        return F.log_softmax(self.classifier(data))
    
    
    def get_adversary(self, attack_type, epsilon, class_num):
        self.classifier = self.pretrained_model(self.model_path)
        
        if attack_type == 'fgsm':
            adversary = GradientSignAttack(
                        self.classifier_adv, loss_fn=nn.CrossEntropyLoss(), eps=epsilon, clip_min=0.0, 
                        clip_max=1.0, targeted=False)
        elif attack_type == 'pgd':
            adversary = PGDAttack(
                        self.classifier_adv, loss_fn=nn.CrossEntropyLoss(), eps=epsilon,
                        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                        targeted=False)
        elif attack_type == 'cw':
            adversary = CW(
                        self.classifier_adv, radius=epsilon, class_num=class_num ,device=self.device)
        else:
            assert False, 'There\' no such attack'
            
        return adversary
    
    def gen_adversary(self, attack_type, epsilon, class_num):
        stime = time.time()

        true_correct = 0
        attacked_correct = 0

        adv_example, org_img ,attack_labels, true_labels, success = [], [], [], [], []
        attack_logit, org_logit = [], []
        
        train_loader, test_loader = self.data_loader()
        adversary = self.get_adversary(attack_type, epsilon, class_num)
        classifier = self.pretrained_model(self.model_path)
        
        for i, (data, target) in enumerate(tqdm(test_loader)):
            data = data.to(self.device)
            target = target.to(self.device).long()
            
            adv_img = adversary.perturb(data.clone().detach(), target)

            if self.dataset == 'cifar10':
                adv_img = normalize(adv_img)
                data = normalize(data)
            
            # Adversarial data predict
            output = classifier(adv_img)
            pred = F.log_softmax(output)
            pred = torch.argmax(pred, 1)
            
            # Original data predict
            true_output = classifier(data)
            true_pred = F.log_softmax(true_output)
            true_pred = torch.argmax(true_pred, 1)
            
            true_correct += (true_pred.detach().cpu().numpy()==target.detach().cpu().numpy()).sum()
            attacked_correct += (pred.detach().cpu().numpy()==target.detach().cpu().numpy()).sum()
            
            if self.dataset == 'cifar10':
                adv_img = denormalize(adv_img)
                data = denormalize(data)
            
            for j in range(data.shape[0]):
                # In correct predicted data
                if true_pred[j].detach().cpu().numpy()==target[j].detach().cpu().numpy():    
                    adv_example.append(adv_img[j].detach().cpu().numpy())
                    org_img.append(data[j].detach().cpu().numpy())
                    attack_labels.append(pred[j].detach().cpu().numpy())
                    true_labels.append(true_pred[j].detach().cpu().numpy())
                    
                    # attack success : 1
                    # attack fail    : 0
                    success.append((pred[j].detach().cpu().numpy()!=target[j].detach().cpu().numpy()).astype('int'))

                    attack_logit.append(torch.max(output[j]).item())
                    org_logit.append(torch.max(true_output[j]).item())

                    
        etime = time.time()
        print('Original Accuracy %.2f' %(true_correct * 100 / len(test_loader.dataset)))
        print('Attacked Accuracy %.2f' %(attacked_correct * 100 / len(test_loader.dataset)))
        
        
        if self.dataset == 'mnist':
            adv_example = np.array(adv_example).reshape(-1, 28, 28)
            org_img = np.array(org_img).reshape(-1, 28, 28)
        elif self.dataset == 'cifar10':
            adv_example = np.array(adv_example).reshape(-1, 3, 32, 32)
            org_img = np.array(org_img).reshape(-1, 3, 32, 32)
        else:
            assert False, 'There\'s no such dataset'
            
        attack_labels = np.array(attack_labels).reshape(-1)
        true_labels = np.array(true_labels).reshape(-1)
        success = np.array(success).reshape(-1)
        attack_logit = np.array(attack_logit).reshape(-1)
        org_logit = np.array(org_logit).reshape(-1)
        
        attack_fail = (success == 0).sum()
        attack_success = (success == 1).sum()
        success_rate = (attack_success / (attack_fail + attack_success)) * 100
        
        print('Correct predict :', (attack_fail + attack_success))
        print('Attack fail :', attack_fail)
        print('Attack success :', attack_success)
        print('Attack success rate : %.2f' % (attack_success * 100 / (attack_fail + attack_success)))
        print('\nElapsed time : %.3f' % (etime-stime))
        
        return adv_example, org_img, attack_labels, true_labels, success, attack_logit, org_logit



def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if type(m.bias) != type(None):
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if type(m.bias) != type(None):
                init.constant_(m.bias, 0)
                


def get_data(dataset='cifar10'):
    if dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
          
        data_train = CIFAR10('./data/cifar10',
                           download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32,32),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, 4),
                               transforms.ToTensor(),
                               normalize,]))
        data_test = CIFAR10('./data/cifar10',
                          train=False,
                          download=True,
                          transform=transforms.Compose([
                          transforms.Resize(32,32),
                          transforms.ToTensor(),
                              ]))
        return data_train,data_test
    
    elif dataset == 'mnist':
        data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor()]))
        data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((28, 28)),
                      transforms.ToTensor()]))
        return data_train, data_test

    
# adversary [img, pred, logit, success]
# original [img, pred, logit]

def save_training_data(train_img, train_label, train_logit, success, is_adversary,
                       dataset, save_path, attack_type, epsilon):
    
    if is_adversary == True:
        gen_data = [train_img, train_label, train_logit, success]
        attack_fail = (success == 0).sum()
        attack_success = (success == 1).sum()
        success_rate = (attack_success / (attack_fail + attack_success)) * 100

        print('Correct predict :', (attack_fail + attack_success))
        print('Attack fail :', attack_fail)
        print('Attack success :', attack_success)
        print('Attack success rate : %.2f' % (attack_success * 100 / (attack_fail + attack_success)))

        file_name = '%s_%s_attacked_eps%.2f_tpr%d.pkl' %(dataset, attack_type, epsilon, round(success_rate) // 1)
        dir_path = os.path.join(save_path, dataset, attack_type)
        path = os.path.join(dir_path, file_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        save_list(gen_data, path)
        print('Adversary saved')

        # remove .pkl
        if not os.path.exists(path[:-4]):
            os.makedirs(path[:-4])
            os.makedirs(os.path.join(path[:-4], 'train'))
            os.makedirs(os.path.join(path[:-4], 'test'))

        # train_test_split
        data = load_list(path)

        # load adversarial data
        img = data[0][data[3] == 1]
        label = data[1][data[3] == 1]
        logit = data[2][data[3] == 1]

        train_X, test_X, train_Y, test_Y, train_L, test_L = train_test_split(img, label, logit, test_size=0.2)

        gen_train_data = [train_X, train_Y, train_L]
        gen_test_data = [test_X, test_Y, test_L]
        print('# of train data :', train_X.shape[0])
        print('# of test data :', test_X.shape[0])

        save_list(gen_train_data, os.path.join(path[:-4], '%s_%s_train_%d.pkl'%(dataset, attack_type, train_X.shape[0])))
        save_list(gen_test_data, os.path.join(path[:-4], '%s_%s_test_%d.pkl'%(dataset, attack_type, test_X.shape[0])))

        print('train_test_split finished')

    else:
        gen_data = [train_img, train_label, train_logit]

        print('Correct predict :', len(success))

        file_name = '%s_org_data.pkl' % dataset
        dir_path = os.path.join(save_path, dataset, 'org')
        path = os.path.join(dir_path, file_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        save_list(gen_data, path)
        print('Original saved')

        # remove .pkl
        if not os.path.exists(path[:-4]):
            os.makedirs(path[:-4])
            os.makedirs(os.path.join(path[:-4], 'train'))
            os.makedirs(os.path.join(path[:-4], 'test'))
        
        data = load_list(path)
        
        # load adversarial data
        img = data[0]
        label = data[1]
        logit = data[2]

        train_X, test_X, train_Y, test_Y, train_L, test_L = train_test_split(img, label, logit, test_size=0.2)
        print('# of train data :', train_X.shape[0])
        print('# of test data :', test_X.shape[0])

        gen_train_data = [train_X, train_Y, train_L]
        gen_test_data = [test_X, test_Y, test_L]

        save_list(gen_train_data, os.path.join(path[:-4], '%s_org_train_%d.pkl'%(dataset, train_X.shape[0])))
        save_list(gen_test_data, os.path.join(path[:-4], '%s_org_test_%d.pkl'%(dataset, test_X.shape[0])))

        print('train_test_split finished')
    
    
def save_list(data, path):
    with open(path, "wb") as fp:   #Pickling
        pickle.dump(data, fp)
    
    
def load_list(path):
    with open(path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        return b
    
def plot(original_imgs, adv_imgs, size, dataset='mnist'):
    fig, ax = plt.subplots(size,size*2, figsize=(20, 14))
    for i in range(size):
        for j in range(size*2):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            if(j<size):
                ax[i][j].title.set_fontsize(17)
                ax[i][j].title.set_text('Org')
                img = original_imgs[i*size+j]
                if dataset == 'cifar10':
                    ax[i][j].imshow(np.transpose(img, (1,2,0)))
                elif dataset == 'mnist':
                    ax[i][j].imshow(img)
            else:
                ax[i][j].title.set_fontsize(17)
                ax[i][j].title.set_text('Adv')
                img = adv_imgs[i*size+j-size]
                if dataset == 'cifar10':
                    ax[i][j].imshow(np.transpose(img, (1,2,0)))
                elif dataset == 'mnist':
                    ax[i][j].imshow(img)

    fig.tight_layout()
  
    
def normalize(data):
    data[:,0,:,:] -= 0.485
    data[:,0,:,:] /= 0.229
    
    data[:,1,:,:] -= 0.456
    data[:,1,:,:] /= 0.224
    
    data[:,2,:,:] -= 0.406
    data[:,2,:,:] /= 0.225
    
    return data

def denormalize(data):
    data[:,0,:,:] *= 0.229
    data[:,0,:,:] += 0.485
    
    data[:,1,:,:] *= 0.224
    data[:,1,:,:] += 0.456
    
    data[:,2,:,:] *= 0.225
    data[:,2,:,:] += 0.406
    
    return data