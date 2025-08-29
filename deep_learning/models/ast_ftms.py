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

class SourceToTargetFrequencyMixStyle(nn.Module):
    def __init__(self, f_dim, num_domains, p=1.0, alpha=1.0, eps=1e-6):
        super().__init__()
        self.f_dim = f_dim
        self.p = p
        self.alpha = alpha  # 혼합 강도 조절 파라미터
        self.eps = eps
        self.num_domains = num_domains
        self._beta = torch.distributions.Beta(self.alpha, self.alpha)
        
    def forward(self, x, domain_labels, block=0):
        # 학습 중이 아니거나 확률 p보다 큰 랜덤 값이 나오면 원본 반환
        if not self.training or torch.rand(1) > self.p:
            return x
    
        
        # 소스 도메인과 타겟 도메인 마스크 생성
        source_mask = (domain_labels == 1)  
        target_mask = (domain_labels == 0)   
        
        # 소스나 타겟 샘플이 없으면 원본 반환
        if source_mask.sum() == 0 and target_mask.sum() == 0:
            return x
        
        # 블록에 따라 텐서 형태 조정
        if block == 0:  # 입력 이미지 형태 (B, C, T, F)(B , 1, 398, 128)
            B, C, t_dim, f_dim = x.shape
            # x = x.view(B, f_dim, t_dim, C)
            
            # 소스 샘플 변환
            x_transformed = x.clone()
            for i in range(B):
                if source_mask[i]:
                    # 타겟 도메인 샘플 중 하나를 랜덤하게 선택
                    target_indices = torch.where(target_mask)[0]
                    if len(target_indices) > 0:
                        j = target_indices[torch.randint(0, len(target_indices), (1,))]
                                                
                        # 선택된 타겟 샘플의 통계 계산
                        target_mu = x[j:j+1].mean(dim=[2], keepdim=True)
                        target_sig = (x[j:j+1].var(dim=[2], keepdim=True) + self.eps).sqrt()
                        
                        source_mu = x[i:i+1].mean(dim=[2], keepdim=True)
                        source_sig = (x[i:i+1].var(dim=[2], keepdim=True) + self.eps).sqrt()
                        
                        # 알파 값으로 혼합 강도 조절
                        lam = self._beta.sample().to(x.device)
                        lam = lam.view(1, 1, 1, 1)
                        
                        mu_mix = (1 - lam) * source_mu + lam * target_mu
                        sig_mix = (1 - lam) * source_sig + lam * target_sig
                        
                        # 정규화 후 새 통계로 변환
                        x_norm = (x[i:i+1] - source_mu) / source_sig
                        x_transformed[i] = x_norm * sig_mix + mu_mix
                
                # elif target_mask[i]:
                #     # 타겟 샘플은 다른 타겟 샘플과만 믹스
                #     other_target_indices = torch.where(target_mask)[0]
                #     # 자기 자신 제외
                #     other_target_indices = other_target_indices[other_target_indices != i]
                    
                #     if len(other_target_indices) > 0:
                #         j = other_target_indices[torch.randint(0, len(other_target_indices), (1,))]
                        
                #         # 선택된 타겟 샘플의 통계 계산
                #         other_mu = x[j:j+1].mean(dim=[2], keepdim=True)
                #         other_sig = (x[j:j+1].var(dim=[2], keepdim=True) + self.eps).sqrt()
                        
                #         target_mu = x[i:i+1].mean(dim=[2], keepdim=True)
                #         target_sig = (x[i:i+1].var(dim=[2], keepdim=True) + self.eps).sqrt()
                        
                #         # 알파 값으로 혼합 강도 조절
                #         lam = self._beta.sample().to(x.device)
                #         lam = lam.view(1, 1, 1, 1)
                        
                #         mu_mix = (1 - lam) * target_mu + lam * other_mu
                #         sig_mix = (1 - lam) * target_sig + lam * other_sig
                        
                #         # 정규화 후 새 통계로 변환
                #         x_norm = (x[i:i+1] - target_mu) / target_sig
                #         x_transformed[i] = x_norm * sig_mix + mu_mix
            
            return x_transformed
                
        else:  # 트랜스포머 블록 내부 형태 (B, N, C)
            B, N, C = x.shape
            t_dim = N // self.f_dim
            x = x.reshape(B, self.f_dim, t_dim, C)
            
            # print('f_dim', self.f_dim) # 39
            # print('t_dim', t_dim) # 12
            # 소스 샘플 변환
            x_transformed = x.clone()
            for i in range(B):
                # 타겟 샘플은 타겟끼리만 믹스, 소스 샘플은 소스와 타겟 모두와 믹스 가능
                if source_mask[i]:
                    # 타겟 도메인 샘플 중 하나를 랜덤하게 선택
                    target_indices = torch.where(target_mask)[0]
                    if len(target_indices) > 0:
                        j = target_indices[torch.randint(0, len(target_indices), (1,))]
                        
                        target_mu = x[j:j+1].mean(dim=[1], keepdim=True)
                        target_sig = (x[j:j+1].var(dim=[1], keepdim=True) + self.eps).sqrt()
                        
                        source_mu = x[i:i+1].mean(dim=[1], keepdim=True)
                        source_sig = (x[i:i+1].var(dim=[1], keepdim=True) + self.eps).sqrt()
                        
                        # 알파 값으로 혼합 강도 조절
                        lam = self._beta.sample().to(x.device)
                        lam = lam.view(1, 1, 1, 1)
                        
                        mu_mix = (1 - lam) * source_mu + lam * target_mu
                        sig_mix = (1 - lam) * source_sig + lam * target_sig
                        
                        # 정규화 후 새 통계로 변환
                        x_norm = (x[i:i+1] - source_mu) / source_sig
                        x_transformed[i] = x_norm * sig_mix + mu_mix
                
                elif target_mask[i]:
                    # 타겟 샘플은 다른 타겟 샘플과만 믹스
                    other_target_indices = torch.where(target_mask)[0]
                    # 자기 자신 제외
                    other_target_indices = other_target_indices[other_target_indices != i]
                    
                    if len(other_target_indices) > 0:
                        j = other_target_indices[torch.randint(0, len(other_target_indices), (1,))]
                        
                        # 선택된 타겟 샘플의 통계 계산
                        other_mu = x[j:j+1].mean(dim=[1], keepdim=True)
                        other_sig = (x[j:j+1].var(dim=[1], keepdim=True) + self.eps).sqrt()
                        
                        target_mu = x[i:i+1].mean(dim=[1], keepdim=True)
                        target_sig = (x[i:i+1].var(dim=[1], keepdim=True) + self.eps).sqrt()
                        
                        # 알파 값으로 혼합 강도 조절
                        lam = self._beta.sample().to(x.device)
                        lam = lam.view(1, 1, 1, 1)
                        
                        mu_mix = (1 - lam) * target_mu + lam * other_mu
                        sig_mix = (1 - lam) * target_sig + lam * other_sig
                        
                        # 정규화 후 새 통계로 변환
                        x_norm = (x[i:i+1] - target_mu) / target_sig
                        x_transformed[i] = x_norm * sig_mix + mu_mix
            
            return x_transformed.reshape(B, N, C)
        
        

class RandomFrequencyMixStyle(nn.Module):
    def __init__(self, f_dim, num_domains, p=0.3, alpha=1.0, eps=1e-6):
        super().__init__()
        self.f_dim = f_dim
        self.p = p
        self.alpha = alpha  # 혼합 강도 조절 파라미터
        self.eps = eps
        self.num_domains = num_domains
        self._beta = torch.distributions.Beta(self.alpha, self.alpha)
        
    def forward(self, x, domain_labels, block=0):
        # 학습 중이 아니거나 확률 p보다 큰 랜덤 값이 나오면 원본 반환
        if not self.training or torch.rand(1) > self.p:
            return x
        
        # 블록에 따라 텐서 형태 조정
        if block == 0:  # 입력 이미지 형태 (B, C, F, T)
            B, C, f_dim, t_dim = x.shape
            
            # 각 샘플에 대해 무작위로 다른 샘플 선택하여 믹스
            x_transformed = x.clone()
            for i in range(B):
                # 현재 샘플과 다른 무작위 샘플 선택
                j = torch.randint(0, B, (1,))
                while j == i:  # 자기 자신이 아닌 다른 샘플 선택
                    j = torch.randint(0, B, (1,))
                
                # 선택된 샘플의 통계 계산
                target_mu = x[j:j+1].mean(dim=[2], keepdim=True)
                target_sig = (x[j:j+1].var(dim=[2], keepdim=True) + self.eps).sqrt()
                
                source_mu = x[i:i+1].mean(dim=[2], keepdim=True)
                source_sig = (x[i:i+1].var(dim=[2], keepdim=True) + self.eps).sqrt()
                
                # 알파 값으로 혼합 강도 조절
                lam = self._beta.sample().to(x.device)
                lam = lam.view(1, 1, 1, 1)
                
                mu_mix = (1 - lam) * source_mu + lam * target_mu
                sig_mix = (1 - lam) * source_sig + lam * target_sig
                
                # 정규화 후 새 통계로 변환
                x_norm = (x[i:i+1] - source_mu) / source_sig
                x_transformed[i] = x_norm * sig_mix + mu_mix
            
            return x_transformed
                
        else:  # 트랜스포머 블록 내부 형태 (B, N, C)
            B, N, C = x.shape
            t_dim = N // self.f_dim
            x = x.reshape(B, self.f_dim, t_dim, C)
            
            # 각 샘플에 대해 무작위로 다른 샘플 선택하여 믹스
            x_transformed = x.clone()
            for i in range(B):
                # 현재 샘플과 다른 무작위 샘플 선택
                j = torch.randint(0, B, (1,))
                while j == i:  # 자기 자신이 아닌 다른 샘플 선택
                    j = torch.randint(0, B, (1,))
                
                # 선택된 샘플의 통계 계산
                target_mu = x[j:j+1].mean(dim=[1,2], keepdim=True)
                target_sig = (x[j:j+1].var(dim=[1,2], keepdim=True) + self.eps).sqrt()
                
                source_mu = x[i:i+1].mean(dim=[1,2], keepdim=True)
                source_sig = (x[i:i+1].var(dim=[1,2], keepdim=True) + self.eps).sqrt()
                
                # 알파 값으로 혼합 강도 조절
                lam = self._beta.sample().to(x.device)
                lam = lam.view(1, 1, 1, 1)
                
                mu_mix = (1 - lam) * source_mu + lam * target_mu
                sig_mix = (1 - lam) * source_sig + lam * target_sig
                
                # 정규화 후 새 통계로 변환
                x_norm = (x[i:i+1] - source_mu) / source_sig
                x_transformed[i] = x_norm * sig_mix + mu_mix
            
            return x_transformed.reshape(B, N, C)

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
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True, mix_beta=None, domain_label_dim=527,device_label_dim=527, random_mix=False):
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
        self.random_mix = random_mix
        
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
        

        if self.random_mix:
            self.fmix1 = RandomFrequencyMixStyle(self.f_dim, 2, p=1.0, alpha=0.5)
        else:
            self.fmix1 = SourceToTargetFrequencyMixStyle(self.f_dim, 2, p=0.5, alpha=0.5)
    


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

  
        
        


    @autocast()
    def forward(self, x, args=None, training=False, frequency_stylemix=False, domain_labels=None, class_labels=None):
        """
        :param x: the input spectrogram, expected shape: (batch_size, 1, time_frame_num, frequency_bins), e.g., (12, 1, 1024, 128)
        :return: prediction
        """
        
        if training:
            # 1. MixStyle 먼저 적용 (왜곡되지 않은 원본 데이터에 대해)
            if frequency_stylemix:

                x = self.fmix1(x, domain_labels, 0)
                
                x = args.transforms(x)
        
       
        # 입력 변환
        x = x.transpose(2, 3)  # [B, 1, F, T]
                
        B = x.shape[0]
        x = self.v.patch_embed(x)
                
                
        
            
        
        # if training and frequency_stylemix:
        #     if domain_labels is not None and class_labels is not None:
        #         x = self.fmix1(x,domain_labels,1)
                
        
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        
 
        x = self.v.pos_drop(x)
        
 
        
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
            # if training and frequency_stylemix and domain_labels is not None and class_labels is not None:
            #     if i in [5]:  # Apply at 1/4, 1/2, and 3/4 points
            #             x_new = self.fmix1(x[:, 2:, :], domain_labels,1)
            #             x = torch.cat((x[:, :2, :], x_new), dim=1)

        x = self.v.norm(x)
        

        x = (x[:, 0] + x[:, 1]) / 2
        
        

        return x