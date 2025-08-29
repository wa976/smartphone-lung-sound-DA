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


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x



class DomainPrompt(nn.Module):
    """도메인별 프롬프트 임베딩을 생성하는 모듈"""
    def __init__(self, embed_dim=768, prompt_length=10, num_domains=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.prompt_length = prompt_length
        self.num_domains = num_domains
        
        # 각 도메인별 프롬프트 임베딩 생성
        self.domain_prompts = nn.Parameter(
            torch.zeros(num_domains, prompt_length, embed_dim)
        )
        # 프롬프트 임베딩 초기화
        trunc_normal_(self.domain_prompts, std=0.02)
        
    def forward(self, domain_labels):
        """
        도메인 라벨에 따라 해당 도메인의 프롬프트 임베딩 반환
        Args:
            domain_labels: 배치의 도메인 라벨 [batch_size]
        Returns:
            domain_prompt_embeds: 도메인별 프롬프트 임베딩 [batch_size, prompt_length, embed_dim]
        """
        batch_size = domain_labels.shape[0]
        # 각 샘플의 도메인 라벨에 해당하는 프롬프트 임베딩 선택
        domain_prompt_embeds = self.domain_prompts[domain_labels]
        return domain_prompt_embeds

class ASTModel(nn.Module):
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
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True, prompt_learning=False, prompt_length=10, freeze_backbone=False):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        # self.v.blocks = nn.ModuleList([CustomTransformerBlock(...) for _ in range(num_blocks)])

        # 프롬프트 학습 관련 설정
        self.prompt_learning = prompt_learning
        self.prompt_length = prompt_length if prompt_learning else 0
        self.freeze_backbone = freeze_backbone

        self.final_feat_dim = 768
        
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
        
        self.freq_attn_weights = nn.Parameter(torch.randn(self.original_embedding_dim, self.f_dim))


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
            self.domain_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim)) # added for domain adapation
            self.device_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            
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

            # 도메인 프롬프트 모듈 초기화 (프롬프트 학습 활성화 시)
            if self.prompt_learning:
                self.domain_prompt = DomainPrompt(
                    embed_dim=self.original_embedding_dim,
                    prompt_length=prompt_length,
                    num_domains=2  # 0: 병원 데이터, 1: 아이폰 데이터
                )
                

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
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False, prompt_learning=False, freeze_backbone=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]  #1024
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            self.domain_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            self.device_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))# added for domain adapation
            
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
        

        # 항상 domain_prompt 속성 초기화 (prompt_learning이 False일 경우 더미 모듈로)
        if prompt_learning:
            self.domain_prompt = DomainPrompt(
                embed_dim=768,  # 기본값 설정
                prompt_length=prompt_length,
                num_domains=2  # 0: 병원 데이터, 1: 아이폰 데이터
            )
        else:
            # 더미 모듈 (파라미터 없음)
            self.domain_prompt = nn.Module()
            
            
        # 프롬프트 학습이 활성화된 경우 위치 인코딩 확장
        if prompt_learning:
            # 기존 위치 인코딩
            original_pos_embed = self.v.pos_embed  # [1, num_tokens, embed_dim]
            
            # 모든 텐서가 같은 디바이스에 있는지 확인
            device = original_pos_embed.device
            
            # CLS, DIST 토큰 위치 인코딩
            cls_pos_embed = original_pos_embed[:, :2, :]
            
            # 패치 위치 인코딩
            patch_pos_embed = original_pos_embed[:, 2:, :]
            
            # 프롬프트 토큰 위치 인코딩 (같은 디바이스에 생성)
            prompt_pos_embed = torch.zeros(1, prompt_length, self.original_embedding_dim, device=device)
            nn.init.normal_(prompt_pos_embed, std=0.02)
            
            # 새 위치 인코딩 결합
            new_pos_embed = torch.cat([cls_pos_embed, prompt_pos_embed, patch_pos_embed], dim=1)
            
            # 위치 인코딩 업데이트
            self.v.pos_embed_new = nn.Parameter(new_pos_embed)

    def _freeze_backbone(self):
        """백본 모델 파라미터 동결"""
        for param in self.v.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        """백본 모델 파라미터 동결 해제"""
        for param in self.v.parameters():
            param.requires_grad = True

    def set_train_mode(self, stage):
        """학습 단계에 따라 모델 설정 변경"""
        if not self.prompt_learning:
            return
            
                
        if stage == 2:
            # Stage 2: 백본 동결, 프롬프트 학습
            self._freeze_backbone()
            self.domain_prompt.requires_grad_(True)
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
        
        
        
        

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

    def apply_gradient_reversal(self, x, alpha=1.0):
        return GradientReversal.apply(x, alpha)
    

    @autocast()
    def forward(self, x, args=None, training=False, domain_labels=None, class_labels=None, alpha=1.0):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :param domain_labels: 도메인 라벨 (프롬프트 학습 시 사용)
        :param training: 학습 모드 여부
        :param stage: 학습 단계 (1 또는 2)
        :return: embedding
        """
        # 입력 변환
        x = x.transpose(2, 3)  # [B, 1, F, T]
        B = x.shape[0]
        
        # 패치 임베딩
        x = self.v.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # CLS 및 DIST 토큰 추가
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        
        
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
    
        # 드롭아웃
        x = self.v.pos_drop(x)
        
        # 트랜스포머 블록 통과
        for blk in self.v.blocks:
            x = blk(x)
        
        # 정규화
        x = self.v.norm(x)
        features = (x[:, 0] + x[:, 1]) / 2

        return features
