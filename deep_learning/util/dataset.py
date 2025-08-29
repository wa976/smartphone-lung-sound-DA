from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import scipy.signal as signal


import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image
import librosa

from .utils import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio,generate_spectrogram
from .augmentation import augment_raw_audio
import torchaudio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter




def get_domain_infor(domain, args):
    
    if domain == 'iphone':
        device_label = 0
    else:
        device_label = 1
 
                   
    return device_label




class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True,iphone_pre=False):
        print(args.data_folder)
        train_data_folder = os.path.join(args.data_folder, 'train')
        test_data_folder = os.path.join(args.data_folder, 'val')
        self.train_flag = train_flag
        
        self.frame_shift = 10
        
        
        if self.train_flag:
            self.data_folder = train_data_folder
        else:
            self.data_folder = test_data_folder
            
        self.transform = transform
        self.args = args
        
        # parameters for spectrograms
        self.sample_rate = self.args.sample_rate
        self.n_mels = self.args.n_mels        
        
        self.class_nums = np.zeros(args.n_cls)
        
        
        self.target_sample_rate = 4000
        
        
        self.data_glob = sorted(glob(self.data_folder+'/*.wav'))
        
        print('Total length of dataset is', len(self.data_glob))
                                    
        # ==========================================================================
        """ convert fbank """
        self.audio_images = []

  
        for index in self.data_glob: 
            _, file_id = os.path.split(index)
            

            audio, sr = torchaudio.load(index)

            
            file_id_tmp = file_id.split('.wav')[0]

            
            label = file_id_tmp.split('-')[2][0]
            if args.device_mode == 'none':
                device_label = 0
            else:
                device_label = file_id_tmp.split('_')[1][0]
                
            patient_label = file_id_tmp.split(')')[0]
            
   
            #  for total
            if label == "N":
                label = 0
                ori_label = 0
            elif label == "C":
                label = 1
                ori_label = 1
            elif label == "W":
                label = 1
                ori_label = 2
            elif label == "B":
                label = 1
                ori_label = 3
            else:
                print(index)
                continue
            
            
            self.class_nums[int(label)] += 1
            

            if device_label == 'I' or device_label == 0:
                domain = 'iphone'
            else:
                domain = 'stethoscope'
                
      
            

            device_label = get_domain_infor(domain,self.args)
            
            
            audio, sr = torchaudio.load(index)
            
            if iphone_pre:
                if device_label == 0:
                    # 1. Resample (tensor 입력, tensor 출력)
                    resampled_audio = self.resample_audio(audio, sr, 4000)
                    
                    
                    # 2. Apply low-pass filter (입력은 tensor, 출력도 tensor)
                    filtered_audio = self.butter_bandpass_filter(resampled_audio)
                    
                   
                    # # 3. Normalize RMS
                    # normalized_audio = self.rms_normalize(filtered_audio)
                    
                    # 4. 최종 결과를 audio에 할당
                    audio = filtered_audio
                    sr = 4000
            else:
                audio = audio
                
          
            
            audio_image = []
 
            image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels,frame_shift=self.frame_shift) 
            
            # image = generate_fbank(self.args, audio, sr, n_mels=self.n_mels,frame_shift=self.frame_shift) 
            
            audio_image.append(image)
            

            self.audio_images.append((audio_image, int(label), device_label,int(ori_label)))
                
                 
                
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
         
         
        if print_flag:
            print('total number of audio data: {}'.format(len(self.data_glob)))
            print('*' * 25)
            print('For the Label Distribution')
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))



    def resample_audio(self, audio, original_sample_rate, target_sample_rate):
 
        # Resample down to target sample rate
        resampler_down = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        audio_resampled = resampler_down(audio)
        
        
        return audio_resampled
    
    
    def butter_lowpass_filter(self, data, cutoff=950, fs=4000, order=8):
        # 1. tensor를 numpy로 변환
        data_np = data.numpy()

        # 2. 필터 설계 및 적용
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered = signal.lfilter(b, a, data_np)
        
        # 3. numpy를 다시 tensor로 변환
        return torch.from_numpy(filtered).float()
    
    def butter_bandpass_filter(self, data, lowcut=20, highcut=1000, fs=4000, order=4):
        # 1. tensor를 numpy로 변환
        data_np = data.numpy().squeeze()  # Remove extra dimensions
        
        # 2. 필터 설계 및 적용
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered = lfilter(b, a, data_np)
        
        # 3. numpy를 다시 tensor로 변환
        return torch.from_numpy(filtered).float().unsqueeze(0)  # Add channel dimension back

    # 4. Normalize RMS
    def rms_normalize(self, y, target_rms=0.05):
        # Convert tensor to numpy if needed
        if isinstance(y, torch.Tensor):
            y_np = y.numpy().squeeze()
            is_tensor = True
        else:
            y_np = y
            is_tensor = False
            
        rms = np.sqrt(np.mean(y_np**2))
        scale_factor = target_rms / (rms + 1e-12)
        normalized = y_np * scale_factor
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            return torch.from_numpy(normalized).float().unsqueeze(0)
        return normalized
    
    
    
    def __getitem__(self, index):

        audio_images, label, device_label,ori_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2],self.audio_images[index][3]

        audio_image = audio_images[0]
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        
        if self.train_flag:
            return audio_image, (torch.tensor(label),torch.tensor(device_label),torch.tensor(ori_label))
        else:
            return audio_image, torch.tensor(label)
        
        

    def __len__(self):
        return len(self.data_glob)