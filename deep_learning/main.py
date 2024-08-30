from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util.icbhi_diffusion_dataset import ICBHIDataset
from util.icbhi_util import get_score
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from models import get_backbone_class, Projector
from method import PatchMixLoss, PatchMixConLoss
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Sampler

import torch
torch.cuda.set_device(0) 


torch.autograd.set_detect_anomaly(True)


class BalancedDomainClassSampler(Sampler):
    def __init__(self, dataset, batch_size, domain_ratios=None, class_ratios=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.domain_labels, self.class_labels = self._get_labels()
        self.indices_per_domain_class = self._get_indices_per_domain_class()
        self.num_domains = len(set(self.domain_labels))
        self.num_classes = len(set(self.class_labels))
        
        if domain_ratios is None:
            self.domain_ratios = {domain: 1 for domain in set(self.domain_labels)}
        else:
            self.domain_ratios = domain_ratios
        
        if class_ratios is None:
            self.class_ratios = {cls: 1 for cls in set(self.class_labels)}
        else:
            self.class_ratios = class_ratios
        
        self.samples_per_batch = self._get_samples_per_batch()
        self.total_iterations = self._calculate_total_iterations()
    
    def _get_labels(self):
        domain_labels = []
        class_labels = []
        for i in range(len(self.dataset)):
            _, labels = self.dataset[i]
            if isinstance(labels, tuple):
                domain_labels.append(labels[1].item())
                class_labels.append(labels[2].item())
            else:
                domain_labels.append(labels.item())
                class_labels.append(labels.item())  # Assuming class label is same as domain label if not provided separately
        return domain_labels, class_labels
    
    def _get_indices_per_domain_class(self):
        indices_per_domain_class = {}
        for idx, (domain, cls) in enumerate(zip(self.domain_labels, self.class_labels)):
            if domain not in indices_per_domain_class:
                indices_per_domain_class[domain] = {}
            if cls not in indices_per_domain_class[domain]:
                indices_per_domain_class[domain][cls] = []
            indices_per_domain_class[domain][cls].append(idx)
        return indices_per_domain_class
    
    def _get_samples_per_batch(self):
        total_domain_ratio = sum(self.domain_ratios.values())
        total_class_ratio = sum(self.class_ratios.values())
        
        samples_per_batch = {}
        for domain in self.domain_ratios:
            samples_per_batch[domain] = {}
            domain_samples = max(1, int(self.batch_size * (self.domain_ratios[domain] / total_domain_ratio)))
            for cls in self.class_ratios:
                samples_per_batch[domain][cls] = max(1, int(domain_samples * (self.class_ratios[cls] / total_class_ratio)))
        
        return samples_per_batch
    
    def _calculate_total_iterations(self):
        min_samples = min(
            len(indices)
            for domain_indices in self.indices_per_domain_class.values()
            for indices in domain_indices.values()
        )
        min_samples_per_batch = min(
            samples
            for domain_samples in self.samples_per_batch.values()
            for samples in domain_samples.values()
        )
        return math.ceil(min_samples / min_samples_per_batch)
    
    def __iter__(self):
        all_indices = []
        domain_class_iterators = {
            domain: {
                cls: iter(np.random.permutation(indices))
                for cls, indices in class_indices.items()
            }
            for domain, class_indices in self.indices_per_domain_class.items()
        }
        
        for _ in range(self.total_iterations):
            batch = []
            for domain, class_samples in self.samples_per_batch.items():
                for cls, num_samples in class_samples.items():
                    iterator = domain_class_iterators[domain][cls]
                    for _ in range(num_samples):
                        try:
                            index = next(iterator)
                            batch.append(index)
                        except StopIteration:
                            # 이 도메인/클래스의 모든 샘플을 사용했으므로 무시합니다.
                            pass
            
            np.random.shuffle(batch)
            all_indices.extend(batch)
        
        return iter(all_indices)
    
    def __len__(self):
        return self.total_iterations * self.batch_size
    
        
def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    parser.add_argument('--two_cls_eval', action='store_true',
                        help='evaluate with two classes')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data/icbhi_dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='lungsound',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--d_cls', type=int, default=0,
                        help='set k-way classification problem for device (meta)')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=8, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0, 
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])
    parser.add_argument('--nospec', action='store_true')

    # model
    parser.add_argument('--model', type=str, default='ast')
    parser.add_argument('--pretrained', action='store_true')
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    # for AST
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')
    # for SSAST
    parser.add_argument('--ssast_task', type=str, default='ft_avgtok', 
                        help='pretraining or fine-tuning task', choices=['ft_avgtok', 'ft_cls'])
    parser.add_argument('--fshape', type=int, default=16, 
                        help='fshape of SSAST')
    parser.add_argument('--tshape', type=int, default=16, 
                        help='tshape of SSAST')
    parser.add_argument('--ssast_pretrained_type', type=str, default='Patch', 
                        help='pretrained ckpt version of SSAST model')

    parser.add_argument('--method', type=str, default='ce')
    parser.add_argument('--adversarial_ft', action='store_true')    
    # Meta Domain CL & Patch-Mix CL loss
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--mix_beta', default=1.0, type=float,
                        help='patch-mix interpolation coefficient')
    parser.add_argument('--time_domain', action='store_true',
                        help='patchmix for the specific time domain')

    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--colar_weight', type=float, default=0.2)
    parser.add_argument('--pdc_weight', type=float, default=0.5)
    parser.add_argument('--ce_weight', type=float, default=1)
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    parser.add_argument('--negative_pair', type=str, default='all',
                        help='the method for selecting negative pair', choices=['all', 'diff_label'])
    parser.add_argument('--target_type', type=str, default='grad_block',
                        help='how to make target representation', choices=['project_flow', 'grad_block1', 'grad_flow1', 'project_block1', 'grad_block2', 'grad_flow2', 'project_block2', 'project_block_all', 'representation_all', 'grad_block', 'grad_flow', 'project_block'])
    
    # Meta for SCL
    parser.add_argument('--device_mode', type=str, default='none',
                        help='the meta information for selecting', choices=['none', 'mixed'])
    
    # TSNE
    parser.add_argument('--visualize_embeddings', action='store_true',
                    help='visualize initial embeddings by domain and class before training')
    
    # ROC
    parser.add_argument('--roc', action='store_true')
    parser.add_argument('--fold_number', type=int, default='0')
    
    # Frequency StyleMix
    parser.add_argument('--frequency_stylemix', action='store_true')
    
    # Balanced Sampler
    parser.add_argument('--balance_sampler', action='store_true')
    
    
    
    
                        
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    if args.method in ['patchmix', 'patchmix_cl']:
        assert args.model in ['ast', 'ssast']
    
    if args.adversarial_ft:
        args.save_folder = os.path.join(args.save_dir, 'aft', args.model_name)
    else:
        args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    if args.adversarial_ft:
        if args.device_mode in ['mixed']: 
            # args.d_cls = 2
            args.d_cls = 2
        else:
            args.d_cls = 2    
            
    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
            if args.adversarial_ft:
                #single
                if args.device_mode == 'mixed':
                    args.device_cls_list = ['Hospital', 'Iphone']
                else:
                    args.device_cls_list = ['Hospital', 'Iphone']
                    
            if args.n_cls == 4:
                args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
            elif args.n_cls == 2:
                args.cls_list = ['normal', 'abnormal']
                
        elif args.class_split == 'diagnosis':
            if args.n_cls == 3:
                args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
            elif args.n_cls == 2:
                args.cls_list = ['healthy', 'unhealthy']
            else:
                raise NotImplementedError
        
    else:
        raise NotImplementedError
    
    if args.n_cls == 0 and args.m_cls !=0:
        args.n_cls = args.m_cls
        args.cls_list = args.meta_cls_list

    return args

def set_loader(args):
    if args.dataset == 'icbhi':        
        args.h = int(args.desired_length * 100 - 2)
        args.w = 128
        #args.h, args.w = 798, 128
        train_transform = [transforms.ToTensor(),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        ##
        train_transform = transforms.Compose(train_transform)
        val_transform = transforms.Compose(val_transform)
        

        train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args,  print_flag=True)
        

        args.class_nums = train_dataset.class_nums
        
        
    else:
        raise NotImplemented    
    
    
    
    
    if args.balance_sampler:
        print("balance")
        # 예: A와 B 도메인의 비율을 2:1로 설정
        domain_ratios = {0: 6, 1: 10}  # 0은 A 도메인, 1은 B 도메인을 나타낸다고 가정
        # class_ratios = {0: 1, 1: 1}  # Assuming 4 classes with equal ratios
        class_ratios = {0: 3, 1: 1, 2:1, 3:1}  # Assuming 4 classes with equal ratios
        # sampler = BalancedDomainSampler(train_dataset, args.batch_size, domain_ratios=domain_ratios)
        sampler = BalancedDomainClassSampler(train_dataset, args.batch_size, domain_ratios=domain_ratios,class_ratios=class_ratios)
        train_loader = torch.utils.data.DataLoader(train_dataset,  batch_size=args.batch_size,
                                                num_workers=args.num_workers, pin_memory=True,drop_last=True, sampler=sampler)
    else:
        print("normal")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    
    
    return train_loader, val_loader, args


        
        
def set_model(args):
    kwargs = {}
    if args.model == 'ast' or 'ast_ftms':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL
        
    elif args.model == 'ssast':
        kwargs['label_dim'] = args.n_cls
        kwargs['fshape'], kwargs['tshape'] = args.fshape, args.tshape
        kwargs['fstride'], kwargs['tstride'] = 10, 10
        kwargs['input_tdim'] = 798
        kwargs['task'] = args.ssast_task
        kwargs['pretrain_stage'] = not args.audioset_pretrained
        kwargs['load_pretrained_mdl_path'] = args.ssast_pretrained_type
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL
       

    model = get_backbone_class(args.model)(**kwargs)
    
    classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
    projector = Projector(model.final_feat_dim, args.proj_dim) if args.method in ['patchmix_cl'] else nn.Identity()
    
    
    if not args.weighted_loss:
        weights = None
        criterion = nn.CrossEntropyLoss()
    else: 
        weights = torch.tensor(args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum()
    
        criterion = nn.CrossEntropyLoss(weight=weights)
        
           
    if args.model not in ['ast', 'ssast', 'ast_ftms'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')

    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']

        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        
        if ckpt.get('classifier', None) is not None:
            print("correct")
           
            classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
        
    if args.method == 'ce':
        
        criterion = [criterion.cuda()]
    elif args.method == 'patchmix':
        
        if args.adversarial_ft:
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda(), PatchMixLoss(criterion=criterion2).cuda()]
        else:
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda()]
    elif args.method == 'patchmix_cl':
        criterion = [criterion.cuda(), PatchMixConLoss(temperature=args.temperature).cuda()]


    
    #print("device_count :" , torch.cuda.device_count())
    #if torch.cuda.device_count() > 1:
        #model = torch.nn.DataParallel(model)
        
    model.cuda()
    
    classifier.cuda()
    projector.cuda()
    
    
    
    optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, projector, criterion, optimizer




def train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):

    
    model.train()
    
    classifier.train()
    projector.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    
    for idx, (images, labels) in enumerate(train_loader):
        
        # data load
        data_time.update(time.time() - end)
         
        images = images.cuda(non_blocking=True)
        
       
        
        

        class_labels = labels[0].cuda(non_blocking=True)
        device_labels = labels[1].cuda(non_blocking=True)
        ori_labels = labels[2].cuda(non_blocking=True)
            
        bsz = class_labels.shape[0] 
        

        
        
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                
                ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]
                lamb = None

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                
                if args.nospec:
                    features = model(images, args=args, alpha=lamb,training=True)
                else:
                    if args.frequency_stylemix:
                        features = model(args.transforms(images), args=args, alpha=lamb, training=True,frequency_stylemix=True,domain_labels=device_labels,class_labels=ori_labels)
                    else:
                        features = model(args.transforms(images), args=args, alpha=lamb, training=True)
                    # print("input shape : ", images.shape)
                    # print("target shape : ", class_labels.shape)
                    # print("outputs shape : ", features.shape)
                    
    
                    
                    
                
                output = classifier(features)
                loss = criterion[0](output, class_labels)
                
       
                    
                    

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], class_labels, topk=(1,))
        # 바꿔야하는지 불확실
        # [acc1], _ = accuracy(output[:bsz], class_labels if args.adversarial_ft else labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
    
        scaler.scale(loss).backward()

        
        scaler.step(optimizer)
        scaler.update()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
               
                classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])
                projector = update_moving_average(args.ma_beta, projector, ma_ckpt[2])

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    
    # debugger.remove_hooks()
    return losses.avg, top1.avg

def plot_and_save_roc_curve(true_labels, predicted_probs, save_path):
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    
    
def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    
    all_labels = []
    all_preds = []
    
    if args.adversarial_ft:
        classifier = classifier[0]

    classifier.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            


            with torch.cuda.amp.autocast():
                features = model(images, args=args, training=False)
                # features = model(images)
                output = classifier(features)
                loss = criterion[0](output, labels)

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            

            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if labels[idx].item() == 0 and preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                    elif labels[idx].item() != 0 and preds[idx].item() > 0:  # abnormal
                        hits[labels[idx].item()] += 1.0

                
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(output.cpu().numpy()[:, 1])     
            sp, se, sc, f1_normal = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'S_p {sp:.3f}\t'
                      'S_e {se:.3f}\t'
                      'Score {sc:.3f}\t'
                      'F1 Score {f1:.3f}'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, sp=sp, se=se, sc=sc,
                       f1=f1_normal))
                
           
    if args.roc:            
        plot_and_save_roc_curve(all_labels, all_preds, os.path.join('./roc_curve', f'roc_jmir_iphone_ft_fold_{args.fold_number}.png'))
        # Save labels and predictions for average ROC calculation
        np.save(os.path.join('./roc_curve', f'jmir_iphone_ft_fold_{args.fold_number}_labels.npy'), all_labels)
        np.save(os.path.join('./roc_curve', f'jmir_iphone_ft_fold_{args.fold_number}_preds.npy'), all_preds)
    

    # 이 부분을 F1 점수 기준으로 변경
    if f1_normal > best_acc[3]:  # best_acc[3]는 F1 점수를 나타냄
        save_bool = True
        best_acc = [sp, se, sc, f1_normal]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f}, F1 Score: {:.2f}'.format(sp, se, sc, f1_normal))
    print(' * Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f}, F1 Score: {:.2f}'.format(best_acc[0], best_acc[1], best_acc[2], best_acc[3]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))
    
    # if sc > best_acc[-2] and se > 5:
    #     save_bool = True
    #     best_acc = [sp, se, sc, f1_normal]
    #     best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    # print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[2]))
    # print(' * F1 Score: {:.2f} (F1 Score: {:.2f})'.format(f1_normal, best_acc[3]))
    # print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool




def visualize_embeddings(model, train_loader, args):
    model.eval()
    original_embeddings = []
    encoded_embeddings = []
    domains = []
    labels = []
    
    with torch.no_grad():
        for idx, (images, target) in enumerate(train_loader):
            if idx % 10 == 0:  # Print progress every 10 batches
                print(f'Processing batch {idx}/{len(train_loader)}')
            
            images = images.cuda(non_blocking=True)
            
            # Store original image data
            original_embeddings.append(images.cpu().view(images.size(0), -1).numpy())
            
            # Extract encoded embeddings
            features = model(images, args=args, training=False)
            
            encoded_embeddings.append(features.cpu().numpy())
            if isinstance(target, list):
                if len(target) > 1:
                    domains.extend(target[1].numpy())  # Assuming target[1] contains domain information
                labels.extend(target[0].numpy())   # Assuming target[0] contains class labels
            else:
                labels.extend(target.numpy())

    original_embeddings = np.vstack(original_embeddings)
    encoded_embeddings = np.vstack(encoded_embeddings)
    
    # Visualize original embeddings
    visualize_and_save(original_embeddings, domains, labels, args, "original")
    
    # Visualize encoded embeddings
    visualize_and_save(encoded_embeddings, domains, labels, args, "encoded")

    print('Visualization completed.')

def visualize_and_save(embeddings, domains, labels, args, prefix):
    print(f'Performing t-SNE for {prefix} embeddings...')
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    print(f'Plotting {prefix} embeddings...')
    # Plot by domain
    plt.figure(figsize=(20, 16))
    if domains:
        unique_domains = np.unique(domains)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_domains)))
        
        for domain, color in zip(unique_domains, colors):
            mask = np.array(domains) == domain
            plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                        c=[color], label=f'Domain {domain}', alpha=0.6, s=10)
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.6, s=10)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f't-SNE visualization of {prefix} embeddings by domain')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_folder, f'{prefix}_domain_embeddings.png'), dpi=300)
    plt.close()
    
    # Plot by class
    plt.figure(figsize=(20, 16))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                    c=[color], label=f'Class {label}', alpha=0.6, s=10)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f't-SNE visualization of {prefix} embeddings by class')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_folder, f'{prefix}_class_embeddings.png'), dpi=300)
    plt.close()




def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    
    
    best_model = None
    if args.dataset == 'icbhi':
        best_acc = [0, 0, 0, 0]  # Specificity, Sensitivity, Score
    
    if not args.nospec:
        print("sepc")
        args.transforms = SpecAugment(args)
    train_loader, val_loader, args = set_loader(args)
    model, classifier, projector, criterion, optimizer = set_model(args)

    # 여기에 시각화 코드를 추가합니다
    if args.visualize_embeddings:
        visualize_embeddings(model, train_loader, args)
        
        
        
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
    print('Checkpoint Name: {}'.format(args.model_name))
     
    if not args.eval:
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        for epoch in range(args.start_epoch, args.epochs+1):
        
            
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            
            loss, acc = train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, acc))
            
            # eval for one epoch
            best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
            
            # save a checkpoint of model and classifier when the best score is updated
            if save_bool:            
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                # print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc[2], epoch))
                print('Best ckpt is modified with F1 = {:.2f} when Epoch = {}'.format(best_acc[3], epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.adversarial_ft else classifier)
                
                        
            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.adversarial_ft else classifier)
            

        # save a checkpoint of classifier with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        classifier[0].load_state_dict(best_model[1]) if args.adversarial_ft else classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.adversarial_ft else classifier)
        
    else:
        print("correct")
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _  = validate(val_loader, model, classifier, criterion, args, best_acc)
        model.eval()  # Set the model to evaluation mode
        
        


    if args.adversarial_ft:
        update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results_aft.json'))
    else:
        update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))
    print('Checkpoint {} finished'.format(args.model_name))
    
if __name__ == '__main__':
    main()

