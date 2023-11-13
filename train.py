# 切patch并且用了dino(3000)，做个abmil任务

import argparse
import glob
from PIL import Image
import random
import numpy as np
import os
import yaml
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

import torch
import torch.utils.data as Dataloader
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, random_split, ConcatDataset

from model import ABMIL, CLAM_MB, TransMIL
# import dsmil as mil


class EYE5Dataset(Dataset):
    def __init__(self, feat_dir):
        # self.bag_list =  glob.glob(os.path.join(feat_dir, '*/*/*.pkl'))
        # self.bag_list = list(self.bag_list)

        label_list = ['0', '1', '2', '3', '4']
        self.bag_list =  glob.glob(os.path.join(feat_dir, '*/real*.pkl'))
        # self.bag_list = [x for x in bag_list if x.split('/')[-2] in label_list]
    
    def __getitem__(self, index):
        embeds = torch.load(self.bag_list[index], map_location=torch.device('cuda:0'))
        data_class = self.bag_list[index].split('/')[-2]
        label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        label = label_dict.get(data_class)

        return embeds, label

    def __len__(self):
        return len(self.bag_list)


class fakeEYE5Dataset(Dataset):
    def __init__(self, feat_dir):
        # self.bag_list =  glob.glob(os.path.join(feat_dir, '*/*/*.pkl'))
        # self.bag_list = list(self.bag_list)

        label_list = ['0', '1', '2', '3', '4']
        self.bag_list =  glob.glob(os.path.join(feat_dir, '*/fake*.pkl'))
        # self.bag_list = [x for x in bag_list if x.split('/')[-2] in label_list]
    
    def __getitem__(self, index):
        embeds = torch.load(self.bag_list[index], map_location=torch.device('cuda:0'))
        data_class = self.bag_list[index].split('/')[-2]
        label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        label = label_dict.get(data_class)

        return embeds, label

    def __len__(self):
        return len(self.bag_list)
    

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser('Argument of 5 Classes')

parser.add_argument('--batch_size', default=16, type=int)              
parser.add_argument('--lr', default=0.0001, type=float)               
parser.add_argument('--num_epoch', default=40, type=int)
parser.add_argument('--gpu', default='0', type=str)    
parser.add_argument('--seed', default=1234, type=int)                  
parser.add_argument('--feat_dir', default='/data_sdb/THUHITCoop/patches_feature_v2', type=str)     
# parser.add_argument('--rgb_dir', default='/data_sdb/THUHITCoop/1. Classification of Myopic Maculopathy/category', type=str)
# parser.add_argument('--heatmap_dir', default='/data_sdb/THUHITCoop/1. Classification of Myopic Maculopathy/heatmap', type=str)
parser.add_argument('--save_dir', default='/data_sdb/THUHITCoop/Final_code/save_dict/abmilnet', type=str)
# parser.add_argument('--complie', default=False, type=bool)


if __name__ == '__main__':

    args = parser.parse_args()

    # hyparameter
    batch_size = args.batch_size
    NUM_CLASSES = 5
    num_epoch = args.num_epoch
    lr = args.lr
    gpu = args.gpu
    seed = args.seed
    save_dir = args.save_dir
    # rgb_dir = args.rgb_dir
    # heatmap_dir = args.heatmap_dir
    
    train_feat = '/data_sdb/THUHITCoop/1. Classification of Myopic Maculopathy/random_train_features'
    valid_feat = '/data_sdb/THUHITCoop/1. Classification of Myopic Maculopathy/random_valid_features'

    # init
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print('---Using GPU {}---'.format(gpu))

    seed_everything(seed)
    print('---Using seed {}---'.format(seed))

    # dataset implement
    real_train_dataset = EYE5Dataset(feat_dir=train_feat)
    fake_train_dataset = fakeEYE5Dataset(feat_dir=train_feat)

    train_ratio = 0
    total_samples = len(fake_train_dataset)
    train_size = int(train_ratio * total_samples)
    other_size = total_samples - train_size

    fake_train_dataset, _ = random_split(fake_train_dataset, [train_size, other_size])
    train_dataset = ConcatDataset([real_train_dataset, fake_train_dataset])

    valid_dataset = EYE5Dataset(feat_dir=valid_feat)
    
    # 统计每个类别的样本数量
    class_counts = torch.zeros(NUM_CLASSES)

    for _, label in train_dataset:
        class_counts[label] += 1
    
    for i in range(NUM_CLASSES):
        print(f'Train class {i}: {class_counts[i]}')

    class_counts = torch.zeros(NUM_CLASSES)

    for _, label in valid_dataset:
        class_counts[label] += 1
    
    for i in range(NUM_CLASSES):
        print(f'Valid class {i}: {class_counts[i]}')


    train_loader = Dataloader.DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
    valid_loader = Dataloader.DataLoader(valid_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)
    
    class_weights = 1.0 / class_counts
    class_weights = class_weights / torch.sum(class_weights)
    class_weights = class_weights.cuda()

    model = ABMIL(dim_in=384, dim_out=5)
    # model = CLAM_MB(n_classes=5)
    # model = TransMIL(n_classes=5)
    # i_classifier = mil.FCLayer(in_size=384, out_size=5).cuda()
    # b_classifier = mil.BClassifier(input_size=384, output_class=5, dropout_v=0, nonlinear=1).cuda()
    # model = mil.MILNet(i_classifier, b_classifier)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:   # multi-gpu
        model = torch.nn.DataParallel(model)
        print('---using multi-gpu---')

    model = model.cuda()

    # optimizer $ criterion
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           betas=(0.5, 0.9),
                           weight_decay=0.005)
    
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()

    save_path = save_dir

    with open(f'{save_path}/bs{batch_size}_abmilnet_random_train_val.csv', 'w') as file:
        file.write(f'epoch, train_loss, train_acc, valid_kappa, valid_f1, valid_specificity, valid_avg\n')

    best_score = 0.0
    # training
    for epoch in range(num_epoch):
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for i, data in enumerate(tqdm(train_loader)):
            feature_inputs = data[0].cuda()
            labels = data[1].cuda()

            # CLAM
            # outputs, _, _, _, results_dict = model(feature_inputs.squeeze(0), label=labels, instance_eval=True)
            # _, preds = torch.max(outputs, 1)
            # loss = 0.7 * criterion(outputs, labels) + 0.3 * results_dict['instance_loss']

            #DSMIL
            # ins_prediction, bag_prediction, _, _ = model(feature_inputs.squeeze(0))
            # max_prediction, _ = torch.max(ins_prediction, 0)        
            # bag_loss = criterion(bag_prediction, labels)
            # max_loss = criterion(max_prediction.unsqueeze(0), labels)
            # loss = 0.5*bag_loss + 0.5*max_loss
            # _, preds = torch.max(bag_prediction, 1)

            outputs = model(feature_inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feature_inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))

        # validating
        with torch.no_grad():
            model.eval()
            y_true = [] 
            y_pred = [] 
            y_prob = []

            for i, data in enumerate(tqdm(valid_loader)):
                feature_inputs = data[0].cuda()
                labels = data[1].cuda()

                # CLAM
                # outputs, _, _, _, _ = model(feature_inputs.squeeze(0), instance_eval=False)
                # _, preds = torch.max(outputs, 1)

                #DSMIL
                # _, outputs, _, _ = model(feature_inputs.squeeze(0))


                outputs = model(feature_inputs)
                _, preds = torch.max(outputs, 1)

                probs = torch.softmax(outputs, 1).cpu().numpy()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs)

            kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
            f1 = f1_score(y_true, y_pred, average='macro')
            specificity = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            avg_score = (kappa + f1 + specificity) / 3
            print('Valid Kappa: {:.4f} F1: {:.4f} Specificity: {:.4f} AVERAGE: {:.4f}'.format(kappa, f1, specificity, avg_score))

            with open(f'{save_path}/bs{batch_size}_abmilnet_random_train_val.csv', 'a') as file:
                file.write(f'{epoch}, {train_loss}, {train_acc}, {kappa}, {f1}, {specificity}, {avg_score}\n')

            # if avg_score > best_score and avg_score > 0.85:
            #     print('Best Model Saving.....')
            #     best_score = avg_score
            #     torch.save(model.state_dict(), f'{save_path}/best_model.pth')
                


    




