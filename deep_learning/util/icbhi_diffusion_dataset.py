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

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio,generate_spectrogram
from .augmentation import augment_raw_audio
import torchaudio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def save_spectrogram(spectrogram, filename):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def get_icbhi_device_infor(icbhi_device, args):
    
    if args.device_mode == 'none':
        device_label = -1
    
    
    elif args.device_mode == 'mixed':
        if icbhi_device == 'L':
            device_label = 0
        elif icbhi_device == 'A':
            device_label = 1
        elif icbhi_device == 'M':
            device_label = 2
        else:
            device_label = 3
    
                   
    return device_label


def get_domain_infor(domain, args):
    
    if domain == 'iphone':
        device_label = 0
    else:
        device_label = 1
 
                   
    return device_label




class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True):
        train_data_folder = os.path.join(args.data_folder, 'training', 'real')
        test_data_folder = os.path.join(args.data_folder, 'test', 'real')
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

  

        for index in self.data_glob: #for the training set, 4142
            _, file_id = os.path.split(index)
            

            
            
            audio, sr = torchaudio.load(index)

            
            file_id_tmp = file_id.split('.wav')[0]

            
            # for atan
            label = file_id_tmp.split('-')[2][0]
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
            
            
                
            # for atan
            if device_label == 'I':
                domain = 'iphone'
            else:
                domain = 'stethoscope'
                
      
            

            # for atan
            device_label = get_domain_infor(domain,self.args)
            
            
            
            audio, sr = torchaudio.load(index)
            

            if device_label == 0:
                resampled_audio = self.resample_audio(audio,sr,4000)
                
                # Apply low-pass filter
                filtered_audio = self.butter_lowpass_filter(resampled_audio)
                filtered_audio_np = filtered_audio.numpy().squeeze()

               
                # # Convert back to torch tensor
                audio = torch.from_numpy(filtered_audio_np).float().unsqueeze(0)
                
          
            
            audio_image = []
            for aug_idx in range(self.args.raw_augment+1):
                if aug_idx > 0:
                    if self.train_flag:
                        audio = augment_raw_audio(np.asarray(audio.squeeze(0)), self.sample_rate, self.args)
                        audio = cut_pad_sample_torchaudio(torch.tensor(audio), self.args)                
                    
                    image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels)


                    audio_image.append(image)
                else:
                    image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels,frame_shift=self.frame_shift) 
                    
                    audio_image.append(image)
            
            if self.args.device_mode =='none':
                self.audio_images.append((audio_image, int(label)))
            else:
                self.audio_images.append((audio_image, int(label), device_label,int(ori_label)))
                
                 
                
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
         
         
        if print_flag:
            print('total number of audio data: {}'.format(len(self.data_glob)))
            print('*' * 25)
            print('For the Label Distribution')
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))



    def resample_audio(self, audio, original_sample_rate, target_sample_rate):
        """
        Resample the audio to the target sample rate and back to the original sample rate.
        
        Parameters:
        - audio: The input audio signal (PyTorch Tensor).
        - original_sample_rate: The original sample rate of the audio signal.
        - target_sample_rate: The target sample rate to resample to.
        
        Returns:
        - The resampled audio signal back at the original sample rate.
        """

        # Resample down to target sample rate
        resampler_down = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        audio_resampled = resampler_down(audio)
        
        
        return audio_resampled
    
    def butter_lowpass_filter(self, data, cutoff=950, fs=4000, order=8):
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.lfilter(b, a, data)
        y_tensor = torch.from_numpy(y).float()
        return y_tensor
    
    
    
    
    
    
    
    
    def __getitem__(self, index):
        if self.args.device_mode == 'none':
            audio_images, label = self.audio_images[index][0], self.audio_images[index][1]
        else:
            audio_images, label, device_label,ori_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2],self.audio_images[index][3]

        if self.args.raw_augment and self.train_flag:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        


            
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        
        if self.train_flag:
            if self.args.device_mode == 'none':
                return audio_image, torch.tensor(label)
            else:
                return audio_image, (torch.tensor(label),torch.tensor(device_label),torch.tensor(ori_label))
        else:
            return audio_image, torch.tensor(label)
        
        

    def __len__(self):
        return len(self.data_glob)