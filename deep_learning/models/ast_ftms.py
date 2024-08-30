import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from copy import deepcopy
from timm.models.layers import to_2tuple,trunc_normal_
from timm.models.vision_transformer import Block
from .functions import ReverseLayerF



# 옵션 1 : cross 도메인할건지 random 할건지 in 도메인 할건지
# 옵션 2 : 주파수에 대해서만 할건지 시간에 대해서만 할건지 둘 다 할건지
# 옵션 3 : 배치 샘플러 클래스 비율
# 옵션 4 : p, alpha
# 옵션 5 : mix 위치
# 옵션 6 : 비정상을 세부클래스로 믹싱
class CrossDomainClassSpecificFrequencyMixStyle(nn.Module):
    def __init__(self, f_dim, num_domains, num_classes, p=0.3, alpha=0.5, eps=1e-6, mix='crossdomain', class_mix=True):
        super().__init__()
        self.f_dim = f_dim
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.num_domains = num_domains
        self.num_classes = num_classes
        self.mix = mix
        self.class_mix = class_mix
    def forward(self, x, domain_labels, class_labels):
        if not self.training or torch.rand(1) > self.p:
            return x
        
        B, N, C = x.shape
        t_dim = N // self.f_dim
        x = x.view(B, self.f_dim, t_dim, C)
        
        mu = x.mean(dim=[1, 2], keepdim=True)
        var = x.var(dim=[1, 2], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig
        
        # print("class_labels : ", class_labels)
        # print("domain_labels : ", domain_labels)
        
        # 클래스 및 도메인 마스크 생성
        class_masks = (class_labels.unsqueeze(1) == class_labels.unsqueeze(0))
        domain_masks = (domain_labels.unsqueeze(1) != domain_labels.unsqueeze(0))
        valid_indices = class_masks & domain_masks
        
        # 각 샘플에 대한 유효한 믹싱 대상 수 계산
        num_valid_per_sample = valid_indices.sum(dim=1)
        
        # 믹싱 인덱스 초기화
        mixing_indices = torch.arange(B, device=x.device)
        
        # 유효한 믹싱 대상이 있는 샘플에 대해서만 믹싱 수행
        for i in range(B):
            if num_valid_per_sample[i] > 0:
                valid_targets = torch.where(valid_indices[i])[0]
                mixing_indices[i] = valid_targets[torch.randint(0, len(valid_targets), (1,))]
        
        # 알파 값 생성
        alpha = torch.rand(B, 1, 1, 1, device=x.device) * self.alpha
        
        # 믹싱 수행
        mu2, sig2 = mu[mixing_indices], sig[mixing_indices]
        
        mu_mix = mu*alpha + mu2 * (1-alpha)
        sig_mix = sig*alpha + sig2 * (1-alpha)
        
        x_mixed = x_normed*sig_mix + mu_mix
        
        return x_mixed.view(B, N, C)
    

    
    
    
    
        
# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        


    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTFTMSModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True, mix_beta=None, domain_label_dim=527,device_label_dim=527):
        super(ASTFTMSModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        # self.v.blocks = nn.ModuleList([CustomTransformerBlock(...) for _ in range(num_blocks)])

        self.final_feat_dim = 768
        self.mix_beta = mix_beta

        
        # original_embedding_dim 설정
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.original_embedding_dim = 192
            elif model_size == 'small224':
                self.original_embedding_dim = 384
            else:  # base224 or base384
                self.original_embedding_dim = 768
        else:
            self.original_embedding_dim = 768

        # f_dim 계산
        self.f_dim, _ = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        

        
        self.fmix1 = CrossDomainClassSpecificFrequencyMixStyle(self.f_dim, 2,2)
        self.fmix2 = CrossDomainClassSpecificFrequencyMixStyle(self.f_dim, 2,2)
    


        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            self.domain_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, domain_label_dim)) # added for domain adapation
            self.device_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, device_label_dim))
            
            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            out_dir = './pretrained_models/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            
            if os.path.exists(os.path.join(out_dir, 'audioset_10_10_0.4593.pth')) == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out=os.path.join(out_dir, 'audioset_10_10_0.4593.pth'))
            
            sd = torch.load(os.path.join(out_dir, 'audioset_10_10_0.4593.pth'), map_location=device)
            # sd = torch.load(os.path.join('./save/icbhi_ast_ce_jmir_ast_stethoscope_fold0/best.pth'), map_location=device)
            audio_model = ASTFTMSModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False, domain_label_dim=527,device_label_dim=527)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]  #1024
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            self.domain_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, domain_label_dim))
            self.device_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, device_label_dim))# added for domain adapation
            
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))   # test input (1,1,128,400) original embedding 1214 stride()
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def square_patch(self, patch, hw_num_patch):
        h, w = hw_num_patch
        B, _, dim = patch.size()
        square = patch.reshape(B, h, w, dim)
        return square

    def flatten_patch(self, square):
        B, h, w, dim = square.shape
        patch = square.reshape(B, h * w, dim)
        return patch

    def patch_mix(self, image, target, target2, da_index, args, time_domain=False, hw_num_patch=None):
        
        if da_index:
            lam = da_index[0]
            index = da_index[1]
        else:
            
            if self.mix_beta > 0:
                lam = np.random.beta(self.mix_beta, self.mix_beta)
            else:
                lam = 1
        
        batch_size, num_patch, dim = image.size()
        device = image.device

        if not da_index:
            index = torch.randperm(batch_size).to(device)
        

        if not time_domain:
            num_mask = int(num_patch * (1. - lam))
            mask = torch.randperm(num_patch)[:num_mask].to(device)

            image[:, mask, :] = image[index][:, mask, :]
            lam = 1 - (num_mask / num_patch)
        else:
            squared_1 = self.square_patch(image, hw_num_patch)
            squared_2 = self.square_patch(image[index], hw_num_patch)

            w_size = squared_1.size()[2]
            num_mask = int(w_size * (1. - lam))
            mask = torch.randperm(w_size)[:num_mask].to(device)

            squared_1[:, :, mask, :] = squared_2[:, :, mask, :]
            image = self.flatten_patch(squared_1)
            lam = 1 - (num_mask / w_size)
        
        
        
        if args.adversarial_ft:
            y_a, y_b = target, target[index]
            y2_a, y2_b = target2, target2[index]
            return image, (y_a, y2_a), (y_b, y2_b), lam, index
        else:
            y_a, y_b = target, target[index]
            return image, y_a, y_b, lam, index
        
        


    @autocast()
    def forward(self, x, y=None, y2=None, da_index=None, patch_mix=False, time_domain=False, args=None, alpha=None,training=False,frequency_stylemix=False, domain_labels=None,class_labels=None):
        """
        :param x: the input spectrogram, expected shape: (batch_size, 1, time_frame_num, frequency_bins), e.g., (12, 1, 1024, 128)
        :return: prediction
        """
        

        x = x.transpose(2, 3) # B, 1, F, T

        h_patch, w_patch = int((x.size()[2] - 16) / 10) + 1, int((x.size()[3] - 16) / 10) + 1
        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        
        # print("domain_labels : ", domain_labels)
        
        # print("class_labels : ", class_labels)
        
        # print("after patch_embed : ", x)
        
        if frequency_stylemix:
            if domain_labels is not None and class_labels is not None:
                 x = self.fmix1(x, domain_labels, class_labels)
        
        # print("after first fsm : ", x)

        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, y2, da_index, args, time_domain=time_domain, hw_num_patch=[h_patch, w_patch])
        
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
 
        x = self.v.pos_drop(x)
        
        # print("after pos drop : ", x)
 
        
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
            if frequency_stylemix:
                if i == len(self.v.blocks) // 2 and domain_labels is not None and class_labels is not None:
                    x_new = self.fmix2(x[:, 2:, :], domain_labels, class_labels)
                    x = torch.cat((x[:, :2, :], x_new), dim=1)
                    # print("after second fsm : ", x)
                        
    
                
        x = self.v.norm(x)
        
        # print("after norm :", x)
        
        x = (x[:, 0] + x[:, 1]) / 2
        
        # print("final :", x)
        
        
        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index