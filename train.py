import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np

class adv_classifier(nn.Module):
    def __init__(self, params):
        super(adv_classifier, self).__init__()
        
        self.input_size = params['input_size']
        self.output_size = params['output_size']
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, int(self.input_size/2)),
            nn.BatchNorm1d(int(self.input_size/2)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(int(self.input_size/2), int(self.input_size/4)),
            nn.BatchNorm1d(int(self.input_size/4)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(int(self.input_size/4), int(self.input_size/8)),
            nn.BatchNorm1d(int(self.input_size/8)),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        
        self.fc2 = nn.Linear(int(self.input_size/8), self.output_size)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
    
    
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data)
        torch.nn.init.kaiming_uniform_(m.bias.data)
        
def adjust_lr(optimizer, learning_rate ,epoch):
    lr = learning_rate * (0.8 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    
    
def train(model,epochs, optimizer, criterion, train_loader, device,epoch, bound=0.5):
    correct = 0
    model.train()
    total_loss = 0
    adjust_lr(optimizer, optimizer.param_groups[0]['lr'], epoch)
    stime = time.time()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
#         predict = torch.argmax(output.detach(), 1).long()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() / data.shape[0]
        
        
        predict = (output >= bound).float()
        correct += (predict.reshape(-1).detach().cpu().numpy()==target.detach().cpu().numpy()).sum()
        
    etime = time.time()
    accuracy = correct/len(train_loader.dataset)
#     train_accs.append(accuracy * 100)
    print('[%d/%d] train_loss = %.5f\t Elapsed time = %.3f\t Train acc : %.3f' %(epoch+1, epochs, total_loss, (etime-stime),accuracy))
    
    
    
    
    
# for epoch in range(epochs):
def train_bound(model, epochs,optimizer, criterion, train_loader,device, epoch ,bound=0.5):
    correct = 0
    model.train()
    total_loss = 0
    adjust_lr(optimizer, optimizer.param_groups[0]['lr'], epoch)
    stime = time.time()

    for data, target in train_loader:
        
        data, target = data.to(device), target.to(device).float()
        target = target.reshape(-1, 1)
        optimizer.zero_grad()
        output = model(data)
#         predict = torch.argmax(output.detach(), 1).long()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() / data.shape[0]
        
        
        predict = (output >= bound).float()
        correct += (predict.reshape(-1).detach().cpu().numpy()==target.reshape(-1).detach().cpu().numpy()).sum()
        
    etime = time.time()
    accuracy = correct/len(train_loader.dataset)
#     accuracy = correct / (10000 * iters)
#     train_accs.append(accuracy * 100)
    print('correct :',correct)
    print('[%d/%d] train_loss = %.5f\t Elapsed time = %.3f\t Train acc : %.3f' %(epoch+1, epochs, total_loss, (etime-stime),accuracy))
    
    
    
def test(model,epochs, criterion, test_loader ,device,bound=0.5):
    confusion_matrix = np.zeros((3,3))
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        pred_list, target_list = 0, 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            target = target.reshape(-1, 1)
            output = model(data)
#             pred = torch.argmax(output, 1)
            loss = criterion(output, target)
            pred = (output >= bound).float()
            
            correct += (pred.detach().reshape(-1).cpu().numpy()==target.reshape(-1).detach().cpu().numpy()).sum()
            for i in range(len(pred)):
                confusion_matrix[int(target.detach().cpu().numpy()[i])][int(pred.detach().cpu().numpy()[i])]+=1
            
            pred_list = pred.detach().cpu().numpy()
            target_list = target.detach().cpu().numpy()
        
        print('Test acc : %.5f' % (correct / len(test_loader.dataset)))
#         test_accs.append(correct / len(test_loader.dataset))
    
        return pred_list, target_list, confusion_matrix