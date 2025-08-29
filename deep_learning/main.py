from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import matplotlib.pyplot as plt
from util.dataset import ICBHIDataset
from util.utils import get_score
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from models import get_backbone_class
from method.balance_sampler import BalancedDomainSampler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score


    
        
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
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--class_split', type=str, default='lungsound',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--d_cls', type=int, default=0,
                        help='set k-way classification problem for device (meta)')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--test_dataset', type=str, default=None,
                        help='path to independent test dataset for learning curve analysis')
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
    parser.add_argument('--method', type=str, default='ce')

    # Meta for SCL
    parser.add_argument('--device_mode', type=str, default='none',
                        help='the meta information for selecting', choices=['none', 'mixed'])
    
    # Frequency StyleMix
    parser.add_argument('--frequency_stylemix', action='store_true')
    parser.add_argument('--random_mix', action='store_true')
    
    # Balanced Sampler
    parser.add_argument('--balance_sampler', action='store_true')

    # iphone pre
    parser.add_argument('--iphone_pre', action='store_true')

    # t-SNE visualization
    parser.add_argument('--tsne', action='store_true',
                        help='generate t-SNE visualization during evaluation')
    parser.add_argument('--tsne_perplexity', type=int, default=30,
                        help='perplexity parameter for t-SNE')
    parser.add_argument('--tsne_n_iter', type=int, default=1000,
                        help='number of iterations for t-SNE')
    parser.add_argument('--tsne_fontsize', type=int, default=18,
                        help='font size for t-SNE plot')

    # Confusion matrix
    parser.add_argument('--confusion', action='store_true',
                        help='save confusion matrix data during evaluation')
    
    # ROC curve
    parser.add_argument('--roc', action='store_true',
                        help='save ROC curve data during evaluation')

    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)


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

            
    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
            
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
        

        if args.iphone_pre == True:
            train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True,iphone_pre=True)
            val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args,  print_flag=True,iphone_pre=True)
        else:
            train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
            val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args,  print_flag=True)
        

        args.class_nums = train_dataset.class_nums
        
        
    else:
        raise NotImplemented    
    
    
    
    
    if args.balance_sampler:
      
        domain_ratios = {0: 4, 1: 28}  # 0은 A 도메인, 1은 B 도메인을 나타낸다고 가정
        
        print("domain_ratios: ", domain_ratios)

        sampler = BalancedDomainSampler(train_dataset, args.batch_size, domain_ratios=domain_ratios)
        train_loader = torch.utils.data.DataLoader(train_dataset,  batch_size=args.batch_size,
                                                num_workers=args.num_workers, pin_memory=True,drop_last=True, sampler=sampler)
    else:
        print("normal")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    # Create test loader for independent test dataset if provided
    test_loader = None
    if args.test_dataset is not None:
        print(f"Loading independent test dataset from: {args.test_dataset}")
        # Create a copy of args for test dataset
        test_args = deepcopy(args)
        test_args.data_folder = args.test_dataset
        
        if args.iphone_pre == True:
            test_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=test_args, print_flag=True, iphone_pre=True)
        else:
            test_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=test_args, print_flag=True)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)
        print(f"Independent test dataset loaded with {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, args


        
        
def set_model(args):
    kwargs = {}
    if args.model == 'ast' or 'ast_ftms':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        if args.model == 'ast_ftms':
            kwargs['random_mix'] = args.random_mix
    model = get_backbone_class(args.model)(**kwargs)
    
   
    classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast','ast_ftms'] else deepcopy(model.mlp_head).cuda()
    
    projector = nn.Identity()

    
    criterion = [nn.CrossEntropyLoss().cuda()]
    
           
    if args.model not in ['ast',  'ast_ftms'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')

    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu', weights_only=False)
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
        

    
    model.cuda()
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
                alpha = None

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)


        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                if args.nospec:
                    if args.frequency_stylemix:
                        features = model(images, args=args, training=True, 
                                                      frequency_stylemix=True, domain_labels=device_labels, 
                                                      class_labels=ori_labels)
                    else:
                        features = model(images, args=args, training=True, domain_labels=device_labels, class_labels=ori_labels)
                    
                else:
                    if args.frequency_stylemix:
                        features= model(images, args=args, training=True, 
                                                      frequency_stylemix=True, domain_labels=device_labels, 
                                                      class_labels=ori_labels)
                    else:
                        features = model(args.transforms(images), args=args, training=True, domain_labels=device_labels, class_labels=ori_labels)
                    
                output = classifier(features)
                loss = criterion[0](output, class_labels)
                

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], class_labels, topk=(1,))
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

    return losses.avg, top1.avg


    
def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None, is_external=False):
    save_bool = False
    model.eval()
    classifier.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls
    
    # For metrics calculation
    all_labels = []
    all_preds = []
    all_probs = []
    all_features = []  # For t-SNE
    
    # For sample-wise output
    sample_outputs = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            
            # Handle different label formats based on dataset
            if isinstance(labels, list) or isinstance(labels, tuple):
                class_labels = labels[0].cuda(non_blocking=True)
                device_labels = labels[1].cuda(non_blocking=True) if len(labels) > 1 else None
                ori_labels = labels[2].cuda(non_blocking=True) if len(labels) > 2 else None
                labels_for_eval = class_labels
            else:
                labels_for_eval = labels.cuda(non_blocking=True)
                device_labels = None
            
            bsz = labels_for_eval.shape[0]
            
            with torch.cuda.amp.autocast():
                if args.method == 'ce':
                    if args.frequency_stylemix:
                        features = model(images, args=args, training=False, frequency_stylemix=True)
                    else:
                        features = model(images, args=args, training=False)
                        
                    output = classifier(features)
                    loss = criterion[0](output, labels_for_eval)

            # Store data for metrics calculation
            probs = torch.nn.functional.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels_for_eval.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
            # Store sample-wise outputs if in eval mode
            if args.eval:
                for i in range(bsz):
                    sample_outputs.append({
                        'true_label': labels_for_eval[i].item(),
                        'predicted_label': preds[i].item(),
                        'probabilities': probs[i].cpu().numpy().tolist()
                    })
            
            # Store features for t-SNE if needed
            if args.eval and args.tsne:
                all_features.append(features.cpu().numpy())
                if device_labels is not None:
                    # Store device labels for coloring in t-SNE
                    if not 'all_device_labels' in locals():
                        all_device_labels = []
                    all_device_labels.append(device_labels.cpu().numpy())

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels_for_eval, topk=(1,))
            top1.update(acc1[0], bsz)

            for idx in range(preds.shape[0]):
                counts[labels_for_eval[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[idx].item() == labels_for_eval[idx].item():
                        hits[labels_for_eval[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if labels_for_eval[idx].item() == 0 and preds[idx].item() == labels_for_eval[idx].item():
                        hits[labels_for_eval[idx].item()] += 1.0
                    elif labels_for_eval[idx].item() != 0 and preds[idx].item() > 0:  # abnormal
                        hits[labels_for_eval[idx].item()] += 1.0

            sp, se, sc, f1_abnormal, acc = get_score(hits, counts)

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
                      'F1 Score {f1:.3f}\t'
                      'Accuracy {acc:.3f}'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, sp=sp, se=se, sc=sc,
                       f1=f1_abnormal, acc=acc))
        
        # Concatenate all batches
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        
        # Calculate AUROC
        if args.n_cls == 2 or args.two_cls_eval:
            from sklearn.metrics import roc_auc_score
            # For binary: use probability of the positive class
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
            # Calculate AUPRC for binary classification
            auprc = average_precision_score(all_labels, all_probs[:, 1])
        else:
            # For multi-class: use one-vs-rest approach
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import label_binarize
            classes = range(args.n_cls)
            binary_labels = label_binarize(all_labels, classes=classes)
            auroc = roc_auc_score(binary_labels, all_probs, average='macro', multi_class='ovr')
            # Calculate AUPRC for multi-class (macro average)
            auprc = average_precision_score(binary_labels, all_probs, average='macro')

    # Update best_acc to include accuracy, AUROC, and AUPRC
    if len(best_acc) < 7:
        best_acc.extend([0] * (7 - len(best_acc)))  # Add placeholders for accuracy, AUROC, and AUPRC
    
    # Decide whether to save based on F1 + AUPRC average (recommended for medical domain)
    if (f1_abnormal + auprc) / 2 > (best_acc[3] + best_acc[6]) / 2:
        save_bool = True
        best_acc = [sp, se, sc, f1_abnormal, acc, auroc, auprc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f}, F1 Score: {:.3f}, Accuracy: {:.2f}, AUROC: {:.3f}, AUPRC: {:.3f}'.format(
        sp, se, sc, f1_abnormal, acc, auroc, auprc))
    print(' * Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f}, F1 Score: {:.3f}, Accuracy: {:.2f}, AUROC: {:.3f}, AUPRC: {:.3f}'.format(
        best_acc[0], best_acc[1], best_acc[2], best_acc[3], best_acc[4], best_acc[5], best_acc[6]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))
    
    # Save visualization data if in eval mode
    if args.eval:
        # Determine file naming based on whether this is external data
        if is_external:
            data_type = "external"
        else:
            data_type = f"fold_{args.test_fold}" if args.test_fold != 'official' else "official_split"
            fold_name = f"fold_{args.test_fold}" if args.test_fold != 'official' else "official_split"
        
        # Save confusion matrix data
        if args.confusion:
            conf_matrix = confusion_matrix(all_labels, all_preds)
            save_path = os.path.join(args.save_folder, f'confusion_matrix_{data_type}.npy')
            np.save(save_path, conf_matrix)
            print(f"Confusion matrix saved to {save_path}")
            
            # Also save labels and predictions for later use
            np.save(os.path.join(args.save_folder, f'true_labels_{data_type}.npy'), all_labels)
            np.save(os.path.join(args.save_folder, f'predictions_{data_type}.npy'), all_preds)
        
        # Save ROC curve data
        if args.roc:
            roc_data = {
                'labels': all_labels,
                'probs': all_probs
            }
            save_path = os.path.join(args.save_folder, f'roc_data_{data_type}.npz')
            np.savez(save_path, **roc_data)
            print(f"ROC curve data saved to {save_path}")
        
        # Always save external test data (regardless of flags) for visualization
        if is_external:
            # Save external labels and predictions for confusion matrix
            np.save(os.path.join(args.save_folder, f'true_labels_external.npy'), all_labels)
            np.save(os.path.join(args.save_folder, f'predictions_external.npy'), all_preds)
            
            # Save external ROC/PRC data
            external_data = {
                'labels': all_labels,
                'probs': all_probs
            }
            save_path = os.path.join(args.save_folder, f'roc_data_external.npz')
            np.savez(save_path, **external_data)
            print(f"External test data saved for visualization: labels, predictions, and probabilities")
        
        # Generate t-SNE visualization (only for internal data, not external)
        if args.tsne and 'all_features' in locals() and not is_external:
            all_features = np.concatenate(all_features)
            
            # Create t-SNE visualization
            print("Generating t-SNE visualization...")
            tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, 
                        max_iter=args.tsne_n_iter, random_state=args.seed)
            tsne_results = tsne.fit_transform(all_features)
            
            # Create DataFrame for plotting
            df = pd.DataFrame()
            df['x'] = tsne_results[:, 0]
            df['y'] = tsne_results[:, 1]
            df['class'] = all_labels
            
            # Add device labels if available
            if 'all_device_labels' in locals() and len(all_device_labels) > 0:
                all_device_labels = np.concatenate(all_device_labels)
                df['device'] = all_device_labels
                
                # Plot with device coloring
                plt.figure(figsize=(10, 8))
                
                # Define colors for devices
                device_colors = ['blue', 'red']
                device_names = args.device_cls_list if hasattr(args, 'device_cls_list') else ['Device 0', 'Device 1']
                
                # Plot each device type with different color
                for i, device in enumerate(np.unique(all_device_labels)):
                    idx = df['device'] == device
                    plt.scatter(df.loc[idx, 'x'], df.loc[idx, 'y'], 
                                c=device_colors[i % len(device_colors)], 
                                label=device_names[i] if i < len(device_names) else f'Device {i}',
                                alpha=0.7, s=20)
                
                plt.legend(fontsize=args.tsne_fontsize)
                plt.title('t-SNE Visualization of Training Data (Setup 3)', fontsize=args.tsne_fontsize+2, fontweight='bold')
                plt.xlabel('t-SNE dimension 1', fontsize=args.tsne_fontsize)
                plt.ylabel('t-SNE dimension 2', fontsize=args.tsne_fontsize)
                plt.tight_layout()
                
                # Save the figure
                save_path = os.path.join(args.save_folder, f'tsne_device_{fold_name}.png')
                plt.savefig(save_path, dpi=600, bbox_inches='tight')
                plt.close()
                print(f"t-SNE visualization by device saved to {save_path}")
            
            # Plot with class coloring
            plt.figure(figsize=(10, 8))
            
            # Define colors for classes
            class_colors = ['green', 'orange', 'purple', 'brown']
            class_names = args.cls_list if hasattr(args, 'cls_list') else [f'Class {i}' for i in range(args.n_cls)]
            
            # Plot each class with different color
            for i, cls in enumerate(np.unique(all_labels)):
                idx = df['class'] == cls
                plt.scatter(df.loc[idx, 'x'], df.loc[idx, 'y'], 
                            c=class_colors[i % len(class_colors)], 
                            label=class_names[i] if i < len(class_names) else f'Class {i}',
                            alpha=0.7, s=20)
            
            plt.legend(fontsize=args.tsne_fontsize)
            plt.title('t-SNE Visualization by Class', fontsize=args.tsne_fontsize+2)
            plt.xlabel('t-SNE dimension 1', fontsize=args.tsne_fontsize)
            plt.ylabel('t-SNE dimension 2', fontsize=args.tsne_fontsize)
            plt.tight_layout()
            
            # Save the figure
            save_path = os.path.join(args.save_folder, f'tsne_class_{fold_name}.png')
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.close()
            print(f"t-SNE visualization by class saved to {save_path}")

        # Save sample-wise outputs if in eval mode
        if args.eval:
            fold_name = f"fold_{args.test_fold}" if args.test_fold != 'official' else "official_split"
            
            # Save sample-wise predictions and probabilities
            save_path = os.path.join(args.save_folder, f'sample_predictions_{fold_name}.json')
            with open(save_path, 'w') as f:
                json.dump(sample_outputs, f, indent=4)
            print(f"Sample-wise predictions saved to {save_path}")
            
            # Also save as CSV for easier analysis
            import csv
            csv_path = os.path.join(args.save_folder, f'sample_predictions_{fold_name}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                header = ['sample_id', 'true_label', 'predicted_label']
                for i in range(args.n_cls):
                    class_name = args.cls_list[i] if hasattr(args, 'cls_list') and i < len(args.cls_list) else f'class_{i}'
                    header.append(f'prob_{class_name}')
                writer.writerow(header)
                
                # Write data
                for i, sample in enumerate(sample_outputs):
                    row = [i, sample['true_label'], sample['predicted_label']]
                    row.extend(sample['probabilities'])
                    writer.writerow(row)
            print(f"Sample-wise predictions also saved as CSV to {csv_path}")

    # Return current epoch metrics for learning curve
    current_metrics = {
        'sp': sp,
        'se': se, 
        'score': sc,
        'f1': f1_abnormal,
        'acc': acc,
        'auroc': auroc,
        'auprc': auprc
    }
    
    return best_acc, best_model, save_bool, current_metrics


def create_learning_curve_plots(learning_curve, save_folder):
    """Create and save learning curve plots"""
    import matplotlib.pyplot as plt
    import os
    
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle('Learning Curves', fontsize=16)
    
    epochs = learning_curve['epoch']
    
    # Plot 1: Loss and Accuracy
    axes[0, 0].plot(epochs, learning_curve['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Train vs Val Accuracy
    axes[0, 1].plot(epochs, learning_curve['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, learning_curve['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training vs Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Sensitivity and Specificity
    axes[0, 2].plot(epochs, learning_curve['val_se'], 'g-', label='Sensitivity', linewidth=2)
    axes[0, 2].plot(epochs, learning_curve['val_sp'], 'orange', label='Specificity', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Score (%)')
    axes[0, 2].set_title('Sensitivity vs Specificity')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Plot 4: F1 Score
    axes[1, 0].plot(epochs, learning_curve['val_f1'], 'purple', label='F1 Score', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 5: AUROC
    axes[1, 1].plot(epochs, learning_curve['val_auroc'], 'brown', label='AUROC', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUROC')
    axes[1, 1].set_title('Area Under ROC Curve')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 6: Overall Score
    axes[1, 2].plot(epochs, learning_curve['val_score'], 'red', label='Overall Score', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score (%)')
    axes[1, 2].set_title('Overall Score')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    # Plot 7: AUPRC
    axes[2, 0].plot(epochs, learning_curve['val_auprc'], 'cyan', label='AUPRC', linewidth=2)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('AUPRC')
    axes[2, 0].set_title('Area Under Precision-Recall Curve')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Plot 8: F1 + AUPRC (Best Model Criterion)
    if 'val_auprc' in learning_curve:
        criterion_values = [(f1 + auprc) / 2 for f1, auprc in zip(learning_curve['val_f1'], learning_curve['val_auprc'])]
        axes[2, 1].plot(epochs, criterion_values, 'magenta', label='(F1 + AUPRC)/2', linewidth=2)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Criterion Score')
        axes[2, 1].set_title('Best Model Criterion (F1 + AUPRC)/2')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
    
    # Hide the last subplot if not used
    axes[2, 2].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_file = os.path.join(save_folder, 'learning_curves.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curve plots saved to {plot_file}")


def create_combined_learning_curve_plots(val_curve, test_curve, save_folder):
    """Create and save combined learning curve plots (validation + test)"""
    import matplotlib.pyplot as plt
    import os
    
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle('Learning Curves: Validation vs Independent Test', fontsize=16)
    
    val_epochs = val_curve['epoch']
    test_epochs = test_curve['epoch']
    
    # Plot 1: Accuracy comparison
    axes[0, 0].plot(val_epochs, val_curve['val_acc'], 'b-', label='Validation Acc', linewidth=2)
    axes[0, 0].plot(test_epochs, test_curve['test_acc'], 'r-', label='Test Acc', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Validation vs Test Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: F1 Score comparison
    axes[0, 1].plot(val_epochs, val_curve['val_f1'], 'b-', label='Validation F1', linewidth=2)
    axes[0, 1].plot(test_epochs, test_curve['test_f1'], 'r-', label='Test F1', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Validation vs Test F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: AUROC comparison
    axes[0, 2].plot(val_epochs, val_curve['val_auroc'], 'b-', label='Validation AUROC', linewidth=2)
    axes[0, 2].plot(test_epochs, test_curve['test_auroc'], 'r-', label='Test AUROC', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUROC')
    axes[0, 2].set_title('Validation vs Test AUROC')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Plot 4: Sensitivity comparison
    axes[1, 0].plot(val_epochs, val_curve['val_se'], 'b-', label='Validation Sensitivity', linewidth=2)
    axes[1, 0].plot(test_epochs, test_curve['test_se'], 'r-', label='Test Sensitivity', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Sensitivity (%)')
    axes[1, 0].set_title('Validation vs Test Sensitivity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 5: Specificity comparison
    axes[1, 1].plot(val_epochs, val_curve['val_sp'], 'b-', label='Validation Specificity', linewidth=2)
    axes[1, 1].plot(test_epochs, test_curve['test_sp'], 'r-', label='Test Specificity', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Specificity (%)')
    axes[1, 1].set_title('Validation vs Test Specificity')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 6: Overall Score comparison
    axes[1, 2].plot(val_epochs, val_curve['val_score'], 'b-', label='Validation Score', linewidth=2)
    axes[1, 2].plot(test_epochs, test_curve['test_score'], 'r-', label='Test Score', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score (%)')
    axes[1, 2].set_title('Validation vs Test Score')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    # Plot 7: AUPRC comparison
    axes[2, 0].plot(val_epochs, val_curve['val_auprc'], 'b-', label='Validation AUPRC', linewidth=2)
    axes[2, 0].plot(test_epochs, test_curve['test_auprc'], 'r-', label='Test AUPRC', linewidth=2)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('AUPRC')
    axes[2, 0].set_title('Validation vs Test AUPRC')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Plot 8: Best Model Criterion comparison (F1 + AUPRC)/2
    if 'val_auprc' in val_curve and 'test_auprc' in test_curve:
        val_criterion = [(f1 + auprc) / 2 for f1, auprc in zip(val_curve['val_f1'], val_curve['val_auprc'])]
        test_criterion = [(f1 + auprc) / 2 for f1, auprc in zip(test_curve['test_f1'], test_curve['test_auprc'])]
        axes[2, 1].plot(val_epochs, val_criterion, 'b-', label='Validation Criterion', linewidth=2)
        axes[2, 1].plot(test_epochs, test_criterion, 'r-', label='Test Criterion', linewidth=2)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Criterion Score')
        axes[2, 1].set_title('Validation vs Test Criterion (F1+AUPRC)/2')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
    
    # Hide the last subplot if not used
    axes[2, 2].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_file = os.path.join(save_folder, 'combined_learning_curves.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined learning curve plots saved to {plot_file}")


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
        best_acc = [0, 0, 0, 0, 0, 0, 0]  # Specificity, Sensitivity, Score, F1, Accuracy, AUROC, AUPRC
        best_test_acc = [0, 0, 0, 0, 0, 0, 0]  # Independent test best scores (tracking only)
        cv_best_test_acc = [0, 0, 0, 0, 0, 0, 0]  # Test performance when CV model is best
    
    # Learning curve tracking
    learning_curve = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_sp': [],
        'val_se': [], 
        'val_score': [],
        'val_f1': [],
        'val_acc': [],
        'val_auroc': [],
        'val_auprc': []
    }
    
    # Test learning curve tracking (for independent test dataset)
    test_learning_curve = {
        'epoch': [],
        'test_sp': [],
        'test_se': [], 
        'test_score': [],
        'test_f1': [],
        'test_acc': [],
        'test_auroc': [],
        'test_auprc': []
    }
    
    if not args.nospec: 
        print("sepc")
        args.transforms = SpecAugment(args)
        
    train_loader, val_loader, test_loader, args = set_loader(args)
    model, classifier, projector, criterion, optimizer = set_model(args)

        
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
            best_acc, best_model, save_bool, current_metrics = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
            
            # Record learning curve data (convert tensors to float)
            learning_curve['epoch'].append(epoch)
            learning_curve['train_loss'].append(float(loss))
            learning_curve['train_acc'].append(float(acc))
            learning_curve['val_sp'].append(float(current_metrics['sp']))
            learning_curve['val_se'].append(float(current_metrics['se']))
            learning_curve['val_score'].append(float(current_metrics['score']))
            learning_curve['val_f1'].append(float(current_metrics['f1']))
            learning_curve['val_acc'].append(float(current_metrics['acc']))
            learning_curve['val_auroc'].append(float(current_metrics['auroc']))
            learning_curve['val_auprc'].append(float(current_metrics['auprc']))
            
            # Evaluate on independent test dataset if available
            if test_loader is not None:
                print(f"Evaluating on independent test dataset at epoch {epoch}...")
                # Don't use best_acc for test evaluation, just get current metrics
                _, _, _, test_metrics = validate(test_loader, model, classifier, criterion, args, [0, 0, 0, 0, 0, 0, 0], is_external=True)
                
                # Update best test performance (tracking only, doesn't affect model saving)
                test_f1_auprc_avg = (test_metrics['f1'] + test_metrics['auprc']) / 2
                best_test_f1_auprc_avg = (best_test_acc[3] + best_test_acc[6]) / 2
                
                if test_f1_auprc_avg > best_test_f1_auprc_avg:
                    best_test_acc = [float(test_metrics['sp']), float(test_metrics['se']), float(test_metrics['score']), 
                                    float(test_metrics['f1']), float(test_metrics['acc']), float(test_metrics['auroc']), 
                                    float(test_metrics['auprc'])]
                    print(f"New best independent test performance at epoch {epoch}!")
                
                # Record test learning curve data (convert tensors to float)
                test_learning_curve['epoch'].append(epoch)
                test_learning_curve['test_sp'].append(float(test_metrics['sp']))
                test_learning_curve['test_se'].append(float(test_metrics['se']))
                test_learning_curve['test_score'].append(float(test_metrics['score']))
                test_learning_curve['test_f1'].append(float(test_metrics['f1']))
                test_learning_curve['test_acc'].append(float(test_metrics['acc']))
                test_learning_curve['test_auroc'].append(float(test_metrics['auroc']))
                test_learning_curve['test_auprc'].append(float(test_metrics['auprc']))
                
                print(f"Test - Sp: {test_metrics['sp']:.2f}, Se: {test_metrics['se']:.2f}, F1: {test_metrics['f1']:.3f}, AUROC: {test_metrics['auroc']:.3f}, AUPRC: {test_metrics['auprc']:.3f}")
                print(f"Best Test - Sp: {best_test_acc[0]:.2f}, Se: {best_test_acc[1]:.2f}, F1: {best_test_acc[3]:.3f}, AUROC: {best_test_acc[5]:.3f}, AUPRC: {best_test_acc[6]:.3f}")
            
            
            # save a checkpoint of model and classifier when the best score is updated
            if save_bool:            
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                # print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc[2], epoch))
                print('Best ckpt is modified with F1 = {:.3f} when Epoch = {}'.format(best_acc[3], epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)
                
                # Record test performance when CV model is best
                if test_loader is not None:
                    cv_best_test_acc = [float(test_metrics['sp']), float(test_metrics['se']), float(test_metrics['score']), 
                                       float(test_metrics['f1']), float(test_metrics['acc']), float(test_metrics['auroc']), 
                                       float(test_metrics['auprc'])]
                    print('Test performance when CV is best - Sp: {:.2f}, Se: {:.2f}, F1: {:.3f}, AUROC: {:.3f}, AUPRC: {:.3f}'.format(
                        cv_best_test_acc[0], cv_best_test_acc[1], cv_best_test_acc[3], cv_best_test_acc[5], cv_best_test_acc[6]))

                
                        
            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier)


        # save a checkpoint of classifier with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best.pth')

        if best_model is not None:
            model.load_state_dict(best_model[0])
            classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier)
        
        # Save learning curve data
        learning_curve_file = os.path.join(args.save_folder, 'learning_curve.json')
        with open(learning_curve_file, 'w') as f:
            json.dump(learning_curve, f, indent=4)
        print(f"Learning curve data saved to {learning_curve_file}")
        
        # Save test learning curve data if available
        if test_loader is not None and len(test_learning_curve['epoch']) > 0:
            test_learning_curve_file = os.path.join(args.save_folder, 'test_learning_curve.json')
            with open(test_learning_curve_file, 'w') as f:
                json.dump(test_learning_curve, f, indent=4)
            print(f"Test learning curve data saved to {test_learning_curve_file}")
            
            # Create combined learning curve plots
            create_combined_learning_curve_plots(learning_curve, test_learning_curve, args.save_folder)
            
            # Print final best performance summary
            print(f"\n{'='*50}")
            print(f"FINAL BEST PERFORMANCE SUMMARY")
            print(f"{'='*50}")
            print(f"Cross-Validation (used for model saving):")
            print(f"  Sp: {best_acc[0]:.2f}, Se: {best_acc[1]:.2f}, Score: {best_acc[2]:.2f}")
            print(f"  F1: {best_acc[3]:.3f}, Acc: {best_acc[4]:.2f}, AUROC: {best_acc[5]:.3f}, AUPRC: {best_acc[6]:.3f}")
            print(f"  CV Criterion (F1+AUPRC)/2: {(best_acc[3]+best_acc[6])/2:.3f}")
            print(f"\nIndependent Test (tracking only):")
            print(f"  Sp: {best_test_acc[0]:.2f}, Se: {best_test_acc[1]:.2f}, Score: {best_test_acc[2]:.2f}")
            print(f"  F1: {best_test_acc[3]:.3f}, Acc: {best_test_acc[4]:.2f}, AUROC: {best_test_acc[5]:.3f}, AUPRC: {best_test_acc[6]:.3f}")
            print(f"  Test Criterion (F1+AUPRC)/2: {(best_test_acc[3]+best_test_acc[6])/2:.3f}")
            print(f"\nTest Performance when CV is Best:")
            print(f"  Sp: {cv_best_test_acc[0]:.2f}, Se: {cv_best_test_acc[1]:.2f}, Score: {cv_best_test_acc[2]:.2f}")
            print(f"  F1: {cv_best_test_acc[3]:.3f}, Acc: {cv_best_test_acc[4]:.2f}, AUROC: {cv_best_test_acc[5]:.3f}, AUPRC: {cv_best_test_acc[6]:.3f}")
            print(f"  CV-Best Test Criterion (F1+AUPRC)/2: {(cv_best_test_acc[3]+cv_best_test_acc[6])/2:.3f}")
            print(f"{'='*50}")
        else:
            # Create learning curve plots (validation only)
            create_learning_curve_plots(learning_curve, args.save_folder)
                
    else:
        print("Running evaluation mode")
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        
        # Evaluate on validation set
        print("\n" + "="*50)
        print("VALIDATION SET EVALUATION")
        print("="*50)
        best_acc, _, _, val_metrics = validate(val_loader, model, classifier, criterion, args, best_acc)
        
        # Evaluate on independent test dataset if available
        if test_loader is not None:
            print("\n" + "="*50)
            print("INDEPENDENT TEST SET EVALUATION")
            print("="*50)
            test_best_acc = [0, 0, 0, 0, 0, 0, 0]  # Initialize for test evaluation
            test_best_acc, _, _, test_metrics = validate(test_loader, model, classifier, criterion, args, test_best_acc, is_external=True)
            
            # Print comparison summary
            print("\n" + "="*60)
            print("EVALUATION SUMMARY COMPARISON")
            print("="*60)
            print("Validation Set:")
            print(f"  Sp: {val_metrics['sp']:.2f}, Se: {val_metrics['se']:.2f}, Score: {val_metrics['score']:.2f}")
            print(f"  F1: {val_metrics['f1']:.3f}, Acc: {val_metrics['acc']:.2f}, AUROC: {val_metrics['auroc']:.3f}, AUPRC: {val_metrics['auprc']:.3f}")
            print(f"  Criterion (F1+AUPRC)/2: {(val_metrics['f1']+val_metrics['auprc'])/2:.3f}")
            print("\nIndependent Test Set:")
            print(f"  Sp: {test_metrics['sp']:.2f}, Se: {test_metrics['se']:.2f}, Score: {test_metrics['score']:.2f}")
            print(f"  F1: {test_metrics['f1']:.3f}, Acc: {test_metrics['acc']:.2f}, AUROC: {test_metrics['auroc']:.3f}, AUPRC: {test_metrics['auprc']:.3f}")
            print(f"  Criterion (F1+AUPRC)/2: {(test_metrics['f1']+test_metrics['auprc'])/2:.3f}")
            print("="*60)
            
            # Store test results for saving
            best_test_acc = [float(test_metrics['sp']), float(test_metrics['se']), float(test_metrics['score']), 
                           float(test_metrics['f1']), float(test_metrics['acc']), float(test_metrics['auroc']), 
                           float(test_metrics['auprc'])]
        
        # Generate t-SNE visualization for training data if requested
        if args.tsne:
            print("Generating t-SNE visualization for training data...")
            generate_train_tsne(train_loader, model, classifier, args)
        


    # Save both CV and independent test results
    if 'best_test_acc' in locals() and test_loader is not None:
        # Save detailed results as separate JSON file
        detailed_results = {
            'cv_best': [float(x) for x in best_acc],
            'test_best': [float(x) for x in best_test_acc],
            'cv_best_test': [float(x) for x in cv_best_test_acc],
            'cv_criterion': float((best_acc[3] + best_acc[6]) / 2),  # F1 + AUPRC
            'test_criterion': float((best_test_acc[3] + best_test_acc[6]) / 2),  # F1 + AUPRC
            'cv_best_test_criterion': float((cv_best_test_acc[3] + cv_best_test_acc[6]) / 2)  # F1 + AUPRC when CV is best
        }
        detailed_results_file = os.path.join(args.save_folder, 'detailed_results.json')
        with open(detailed_results_file, 'w') as f:
            json.dump(detailed_results, f, indent=4)
        print(f"Detailed results saved to {detailed_results_file}")
        
        # Save CV results to main results.json (for compatibility)
        results_to_save = [float(x) for x in best_acc]
    else:
        results_to_save = [float(x) for x in best_acc]
    
    update_json('%s' % args.model_name, results_to_save, path=os.path.join(args.save_dir, 'results.json'))
    print('Checkpoint {} finished'.format(args.model_name))
    
def generate_train_tsne(train_loader, model, classifier, args):
    """Generate t-SNE visualization for training data with domain labels highlighted."""
    model.eval()
    classifier.eval()
    
    all_features = []
    all_device_labels = []
    all_class_labels = []
    
    # Collect features and labels
    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):
            if idx >= 400:  # Limit to 50 batches to avoid memory issues
                break
                
            images = images.cuda(non_blocking=True)
            
            # Handle different label formats
            if isinstance(labels, list) or isinstance(labels, tuple):
                class_labels = labels[0]
                device_labels = labels[1] if len(labels) > 1 else None
            else:
                class_labels = labels
                device_labels = None
            
            # Extract features
            with torch.cuda.amp.autocast():
                if args.frequency_stylemix:
                    features = model(images, args=args, training=False, frequency_stylemix=True)
                else:
                    features = model(images, args=args, training=False)
            
            all_features.append(features.cpu().numpy())
            all_class_labels.append(class_labels.cpu().numpy())
            
            if device_labels is not None:
                all_device_labels.append(device_labels.cpu().numpy())
    
    # Concatenate all batches
    all_features = np.concatenate(all_features)
    all_class_labels = np.concatenate(all_class_labels)
    
    has_device_labels = len(all_device_labels) > 0
    if has_device_labels:
        all_device_labels = np.concatenate(all_device_labels)
    
    # Create t-SNE visualization
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, 
                max_iter=args.tsne_n_iter, random_state=args.seed)
    tsne_results = tsne.fit_transform(all_features)
    
    # Create DataFrame for plotting
    df = pd.DataFrame()
    df['x'] = tsne_results[:, 0]
    df['y'] = tsne_results[:, 1]
    df['class'] = all_class_labels
    
    fold_name = f"fold_{args.test_fold}" if args.test_fold != 'official' else "official_split"
    
    # Increase font size for all plots
    plt.rcParams.update({'font.size': args.tsne_fontsize + 4})
    
    # If device labels are available, create visualization by device
    if has_device_labels:
        df['device'] = all_device_labels
        
        plt.figure(figsize=(12, 10))  # Increased figure size
        
        # Define colors for devices
        device_colors = ['blue', 'red']
        device_names = ['Smartphone', 'Stethoscope']
        
        # Plot each device type with different color
        for i, device in enumerate(np.unique(all_device_labels)):
            idx = df['device'] == device
            plt.scatter(df.loc[idx, 'x'], df.loc[idx, 'y'], 
                        c=device_colors[i % len(device_colors)], 
                        label=device_names[i] if i < len(device_names) else f'Device {i}',
                        alpha=0.7, s=80)  # Increased point size from 20 to 80
        
        plt.legend(fontsize=args.tsne_fontsize + 6, markerscale=2)  # Increased legend font size and marker size
        plt.title('t-SNE Visualization of Training Data (Setup 3)', fontsize=args.tsne_fontsize + 8, fontweight='bold')
        plt.xlabel('t-SNE dimension 1', fontsize=args.tsne_fontsize + 6)
        plt.ylabel('t-SNE dimension 2', fontsize=args.tsne_fontsize + 6)
        plt.xticks(fontsize=args.tsne_fontsize + 4)  # Increased tick font size
        plt.yticks(fontsize=args.tsne_fontsize + 4)  # Increased tick font size
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(args.save_folder, f'train_tsne_device_{fold_name}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Training data t-SNE visualization by device saved to {save_path}")
    
    # Create visualization by class
    plt.figure(figsize=(12, 10))  # Increased figure size
    
    # Define colors for classes
    class_colors = ['green', 'orange', 'purple', 'brown']
    class_names = args.cls_list if hasattr(args, 'cls_list') else [f'Class {i}' for i in range(args.n_cls)]
    
    # Plot each class with different color
    for i, cls in enumerate(np.unique(all_class_labels)):
        idx = df['class'] == cls
        plt.scatter(df.loc[idx, 'x'], df.loc[idx, 'y'], 
                    c=class_colors[i % len(class_colors)], 
                    label=class_names[i] if i < len(class_names) else f'Class {i}',
                    alpha=0.7, s=80)  # Increased point size from 20 to 80
    
    plt.legend(fontsize=args.tsne_fontsize + 6, markerscale=2)  # Increased legend font size and marker size
    plt.title('t-SNE Visualization of Training Data by Class', fontsize=args.tsne_fontsize + 8)
    plt.xlabel('t-SNE dimension 1', fontsize=args.tsne_fontsize + 6)
    plt.ylabel('t-SNE dimension 2', fontsize=args.tsne_fontsize + 6)
    plt.xticks(fontsize=args.tsne_fontsize + 4)  # Increased tick font size
    plt.yticks(fontsize=args.tsne_fontsize + 4)  # Increased tick font size
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(args.save_folder, f'train_tsne_class_{fold_name}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Training data t-SNE visualization by class saved to {save_path}")
    
    # Reset font size to default
    import matplotlib
    plt.rcParams.update({'font.size': matplotlib.rcParamsDefault['font.size']})

if __name__ == '__main__':
    main()

