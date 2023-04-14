# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:38:36 2022

@author: ljeantet
"""





from Convertisseur_spectro import *
import params

import glob, os
import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from random import randint
import tensorflow_io as tfio
import json



n_fft = params.n_fft # Hann window length
hop_length = params.hop_length
n_mels=params.n_mels
nyquist_rate =params.nyquist_rate
lowpass_cutoff =params.lowpass_cutoff
win_length=params.win_length
f_min = params.f_min # Spectrogram, minimum frequency for call
f_max=params.f_max

type_spec=params.type_spec
sample_rate = params.downsample_rate


conv_spectro=convertisseur_spectro(type_spec, n_fft, hop_length, n_mels, 
                               f_min, f_max, nyquist_rate, win_length )




class Balance_and_data_augmentation:
        def __init__(self, folder,X_file_name,  X_meta_file_name, Y_file_name, out_dir,
                          nb_sgts_ended=200, reduce=True, augmentation=True ):

            
            self.folder=folder
            self.X_file_name=X_file_name
            self.X_meta_file_name=X_meta_file_name
            self.Y_file_name=Y_file_name
            self.out_dir=out_dir
        
            self.nb_sgts_ended=nb_sgts_ended
            self.reduce=reduce
            self.augmentation=augmentation
        
        
                   
    
        def load_data_from_pickle(self):
            '''
            Load all of the spectrograms from a pickle file
                 
            '''
            infile = open(os.path.join(self.folder, self.X_file_name),'rb')
            X = pickle.load(infile)
            infile.close()
                 
            infile = open(os.path.join(self.folder, self.X_meta_file_name),'rb')
            X_meta = pickle.load(infile)
            infile.close()
                 
                 
            infile = open(os.path.join(self.folder, self.Y_file_name),'rb')
            Y = pickle.load(infile)
            infile.close()

            return X, X_meta, Y
     
             
        def load_dico(self, name, key_int=False, print_me=False):
            
            with open(self.folder+name) as f:
                 dico_str = json.loads(json.load(f))
             
            if key_int: 
                conv_key=lambda k:int(k)
            else:
                conv_key=lambda k:k
                 
            dico={conv_key(k):v for k,v in dico_str.items()}
             
            if print_me:
                print(dico)
                 
            return dico  
     
        def time_shifting(self, X, X_meta, index):
            '''
            Augment a segment of amplitude values a number of times (pre-defined).
            Augmenting is done by applying a time shift.
            '''
        
            index=list(index)
           
            idx_pickup=random.sample(index, 1)
        
            segment=X[idx_pickup][0]
            meta=X_meta[idx_pickup][0]

        
            # Randomly select amount to shift by
            random_time_point_segment = randint(1, len(segment)-1)

            # Time shift
            segment = self.time_shift(segment, random_time_point_segment)

            # Append the augmented segments
            
            return segment, meta


        def time_shift(self, audio, shift):
            '''
            Shift ampltitude values (to the right) by a random value.
            Values are wrapped back to the left.
            '''
            
            augmented = np.zeros(len(audio))
            augmented [0:shift] = audio[-shift:]
            augmented [shift:] = audio[:-shift]
            
            return augmented


        def combining_same_class(self, X, X_meta, index):
        
        
            index=list(index)
        
        
            idx_pickup=random.sample(index, 1)
            index.remove(idx_pickup)
            idx_combining=random.sample(index, 1)
        
            #vector to remember which segment has beenn picked up previously 
            segment=self.blend(X[idx_pickup][0], X[idx_combining][0], 0.6, 0.4)
            
            meta=X_meta[idx_pickup][0]   
     
            return segment , meta      
            
        def blend(self, audio_1, audio_2, w_1, w_2):
            augmented = w_1 * audio_1 + w_2 * audio_2
            return augmented 

        
        def add_noise_gaussian(self, X,X_meta,index):
        
            index=list(index)
           
            idx_pickup=random.sample(index, 1)
        
            segment=X[idx_pickup][0]
        
            meta=X_meta[idx_pickup][0]
        
            segment=segment+ 0.009*np.random.normal(0,1,len(segment))
            

            return segment, meta


        def implement_time_mask(self, X,X_meta, index ):
        
            index=list(index)
           
            idx_pickup=random.sample(index, 1)
        
            segment=X[idx_pickup][0]
            meta=X_meta[idx_pickup][0]
        
            spectro=conv_spectro.convert_single_to_image(segment, sample_rate)
              
        
            time_mask = tfio.audio.time_mask(spectro, param=100)

            
            return time_mask.numpy(), meta

        def implement_freq_mask(self, X, X_meta, index ):
        
            index=list(index)
           
            idx_pickup=random.sample(index, 1)
        
            segment=X[idx_pickup][0]
            meta=X_meta[idx_pickup][0]
            spectro=conv_spectro.convert_single_to_image(segment, sample_rate)
        
        
            time_mask = tfio.audio.freq_mask(spectro, param=100)
           
            
            return time_mask.numpy(), meta

        
        def save_data_to_pickle(self, X, X_meta, Y, Saved_X='X_Picidae-pow', Saved_meta='X_meta_Picidae-pow',Saved_Y='Y_Picidae-pow'):
                '''
                Save all of the spectrograms to a pickle file.
        
                '''
                outfile = open(os.path.join(self.out_dir, Saved_X+'.pkl'),'wb')
                pickle.dump(X, outfile, protocol=4)
                outfile.close()
        
                outfile = open(os.path.join(self.out_dir, Saved_meta+'.pkl'),'wb')
                pickle.dump(X_meta, outfile, protocol=4)
                outfile.close()
        
                outfile = open(os.path.join(self.out_dir, Saved_Y+'.pkl'),'wb')
                pickle.dump(Y, outfile, protocol=4)
                outfile.close()  
            
        def Generate_frequency_table(self, X, X_meta, Y, dico ):
    
            meta=[x for x in X_meta]
            col_names=['lat','lon','date','time','cnt']
            Meta=pd.DataFrame(meta, columns=col_names)
            Meta['Label']=Y.copy()
            Meta['Exact_Label']=[dico[str(x)] for x in Meta.Label]

            #explore data
            freq_df=[]
            for species in np.unique(Meta.Exact_Label):
        
                data=Meta[Meta.Exact_Label==species]
                ctn, count= np.unique(data.cnt, return_counts=True)
                for i,j in zip(ctn, count):
                    freq_df.append([species, i, j])
               #print(species, i, j)

            col_name=['Species','Country','freq']   
            freq_db=pd.DataFrame(freq_df, columns=col_name )
        
            return freq_db
    
        def Balance_augment_and_reduce(self, X, X_meta, Y, dico, freq_db, sample_rate):
  

            X_spect=[]
            X_meta_spect=[]
            Y_spect=[]

            for i in range(freq_db.shape[0]):
                species=freq_db.Species[i]
                country=freq_db.Country[i]
                freq=freq_db.freq[i]
            
                Y_labels=[dico[x.astype(str)] for x in Y]
                Meta_cnt=[x[4] for x in X_meta]

            
                if ((freq > self.nb_sgts_ended) & (self.reduce==True)) :
                    print(species, country)
                    print("reduction")
                    index=np.where((np.asarray(Y_labels)==species) & (np.asarray(Meta_cnt)==country))[0]
                    #nb_to_remove=freq-nb_sgts_ended
                    #index_to_remove=np.array(random.sample(list(index), nb_to_remove))
                    index_to_keep=np.array(random.sample(list(index), self.nb_sgts_ended))
                
                    X_spect.extend(conv_spectro.convert_all_to_image(X[index_to_keep], sample_rate))
                    X_meta_spect.extend(X_meta[index_to_keep])
                    Y_spect.extend(Y[index_to_keep])
                
                
                    X=np.delete(X, index, axis=0)
                    X_meta=np.delete(X_meta, index, axis=0 )
                    Y=np.delete(Y, index, axis=0 )
                
                if ((freq < self.nb_sgts_ended) & (self.augmentation==True)):
                    index=np.where((np.asarray(Y_labels)==species) & (np.asarray(Meta_cnt)==country))[0]
                    nb_to_add=self.nb_sgts_ended-freq
                
                    nb_to_augm_per_method=(nb_to_add//5)+1
                
                    X_spect.extend(conv_spectro.convert_all_to_image(X[index], sample_rate))
                    X_meta_spect.extend(X_meta[index])
                    Y_spect.extend(Y[index])
                
                
                    for j in range(0,nb_to_augm_per_method):
                    
                        segment, meta= self.time_shifting(X, X_meta, index)
                        X_spect.append(conv_spectro.convert_single_to_image(segment, sample_rate))
                        X_meta_spect.append(meta)
                    
                        segment, meta= self.combining_same_class(X, X_meta, index)
                        X_spect.append(conv_spectro.convert_single_to_image(segment, sample_rate))
                        X_meta_spect.append(meta)
                    
                        segment, meta= self.add_noise_gaussian(X, X_meta, index)
                        X_spect.append(conv_spectro.convert_single_to_image(segment, sample_rate))
                        X_meta_spect.append(meta)
                    
                        spectro, meta=self.implement_time_mask(X, X_meta, index)
                        X_spect.append(spectro)
                        X_meta_spect.append(meta)
                    
                        spectro, meta=self.implement_freq_mask(X, X_meta, index)
                        X_spect.append(spectro)
                        X_meta_spect.append(meta)
                     
                    Y_spect.extend((5*nb_to_augm_per_method)*[Y[index[0]]])
                
                    X=np.delete(X, index, axis=0)
                    X_meta=np.delete(X_meta, index, axis=0 )
                    Y=np.delete(Y, index, axis=0 )
                    
                    
            return X_spect, X_meta_spect, Y_spect    

            



