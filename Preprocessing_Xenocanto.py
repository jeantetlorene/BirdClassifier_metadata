# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:34:41 2022

@author: ljeantet
"""

import glob, os
import numpy as np
import random
import librosa.display
import librosa
from xml.dom import minidom
from scipy import signal
from random import randint
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import math
import datetime
from os import listdir
from os.path import isfile, join
import os
import json



class Preprocessing_Xenocanto:
    
    def __init__(self, folder, folder_annotation, out_dir, database_file, lowpass_cutoff, 
                 downsample_rate,  segment_duration, hop_chunck, nb_augmented_noise,  
                 nb_augmented_pitch, type_spec, type_saved_data,                 
                 n_fft, hop_length, n_mels, f_min, f_max, nyquist_rate, win_length ):
        self.folder = folder
        self.folder_annotation=folder_annotation
        self.database_file=database_file
        self.out_dir=out_dir

        self.segment_duration = segment_duration
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.hop_chunck=hop_chunck
        self.nb_augmented_noise=nb_augmented_noise
        self.nb_augmented_pitch=nb_augmented_pitch
        self.type_spec=type_spec
        self.type_saved_data=type_saved_data
        self.nyquist_rate = nyquist_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.win_length=win_length

        



    def butter_lowpass(self, cutoff, nyq_freq, order=4):
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype='lowpass')
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
        b, a = self.butter_lowpass(cutoff_freq, nyq_freq, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def read_audio_file(self, file_path):
        '''
       file_name: string, name of file including extension, e.g. "audio1.wav"
    
        '''
   
        # Read the amplitudes and sample rate
        audio_amps, audio_sample_rate = librosa.load(file_path, sr=None)
    
        return audio_amps, audio_sample_rate


    def downsample_file(self, amplitudes, original_sr, new_sample_rate):
        '''
        Downsample an audio file to a given new sample rate.
        amplitudes:
        original_sr:
        new_sample_rate:
        
        '''
        return librosa.resample(amplitudes, 
                                original_sr, 
                                new_sample_rate, 
                                res_type='kaiser_fast'), new_sample_rate


    def build_mel_spectro(self, audio):
        '''
        Convert amplitude values into a mel-spectrogram.
        '''
        S = librosa.feature.melspectrogram(audio, n_fft=self.n_fft,hop_length=self.hop_length,win_length=self.win_length, 
                                           n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        S = librosa.feature.melspectrogram(audio, n_fft=self.n_fft,hop_length=self.hop_length, 
                                           n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        
        image = librosa.core.power_to_db(S)
        image_np = np.asmatrix(image)
        image_np_scaled_temp = (image_np - np.min(image_np))
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps=1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled
        
    
        # 3 different input
        return S1

    def build_spectro(self, audio):
        '''
        Convert amplitude values into a mel-spectrogram.
        '''
        D=librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        image=librosa.amplitude_to_db(abs(D))         
        
 
        image_np = np.asmatrix(image)
        image_np_scaled_temp = (image_np - np.min(image_np))
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps=1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled
        
    
        # 3 different input
        return S1
    
    def build_pcen(self, audio, sample_rate):
        '''
        Convert amplitude values into a mel-spectrogram.
        '''
        
        
        audio = (audio * (2 ** 31)).astype("float32")
        stft = librosa.stft(audio, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        abs2_stft = (stft.real * stft.real) + (stft.imag * stft.imag)
        melspec = librosa.feature.melspectrogram(y=None,S=abs2_stft,sr=sample_rate,n_fft=self.n_fft,hop_length=self.hop_length, 
                                           n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)

        pcen = librosa.pcen(melspec,sr=sample_rate, hop_length=self.hop_length, gain=0.8, bias=10, power=(1/4), time_constant=0.4, 
                     eps=1e-06)

        pcen = pcen.astype("float32")
        
        
    
        # 3 different input
        return pcen
    

    def convert_single_to_image(self,segment, sample_rate):
    
        if self.type_spec=='spectro':
            image=self.build_spectro(segment)
        elif self.type_spec=='mel-spectro':
            image=self.build_mel_spectro(segment)
        elif self.type_spec=='pcen':
            image=self.build_pcen(segment, sample_rate)
        else :
            print('error')
    
        return image 


    
    def convert_all_to_image(self ,audio_list, sample_rate):
        '''
        Convert a number of segments into their corresponding spectrograms.
        '''
        spectrograms = []
        for segment in audio_list:
            spectrograms.append(self.convert_single_to_image(segment, sample_rate))

        return np.array(spectrograms)    


    def print_spectro(self, spectro, sample_rate, title="spectogram"):
        fig, ax = plt.subplots(figsize=(12,5))
        img=librosa.display.specshow(spectro,sr=sample_rate,hop_length=self.hop_length, cmap='magma', x_axis='time',ax=ax)
        fig.colorbar(img, ax=ax,format='%+2.0f dB')
        fig.suptitle(title)



    def get_meta_data(self, file, database_ref):
    
        id_xeno=file.split('-')[0]
        species=file.split('-')[1]
        
        if id_xeno in database_ref['ID'].values:
        
            meta=database_ref[(database_ref['ID']==id_xeno) & (database_ref['Species']==species)][['lat','lng','date','time','cnt']].values
        
        else:
            meta=np.full((1,5), np.nan)
        
        
        return meta[0]


    def create_and_save_dictionnary(self,labels):
    
        labelName_to_labelInd={}
        labelInd_to_labelName={}
    
        for i,name in enumerate(labels):
            labelName_to_labelInd[name]=i
            labelInd_to_labelName[i]=name
    
        #save dictionnary
        with open(self.out_dir+"/labelName_to_labelInd.json", 'w') as f:
            json.dump(json.dumps(labelName_to_labelInd), f)
    
        with open(self.out_dir+"/labelInd_to_labelName.json", 'w') as f:
            json.dump(json.dumps(labelInd_to_labelName), f)
    
        return labelName_to_labelInd,labelInd_to_labelName
    
  

    def load_dico(self, path, key_int=False,print_me=False):
        with open(path) as f:
            dico_str = json.loads(json.load(f))
    
        if key_int: 
            conv_key=lambda k:int(k)
        else:
            conv_key=lambda k:k
        
        dico={conv_key(k):v for k,v in dico_str.items()}
    
        if print_me:
            print(dico)
        
        return dico
    

    
    def pitch_shifting(self, audio_list, sample_rate):

        augmented=[]
        for audio in audio_list:
            for i in range(self.nb_augmented_pitch):
                shift=randint(-5, 5)
                while shift==0:
                    shift=randint(-5, 5)
                    wav_pitch_sf = librosa.effects.pitch_shift(audio,sample_rate,n_steps=shift)
                    augmented.append(wav_pitch_sf)
     
        return augmented   



    def blend(self, audio_1, audio_2, w_1, w_2):
        augmented = w_1 * audio_1 + w_2 * audio_2
        return augmented

    def add_noise(self, audio_list, sample_rate):
    
        fichiers_noise=[f for f in listdir(self.file_path_noise)]
    
        augmented=[]
    
        for audio in audio_list :
        
            for i in range(self.nb_augmented_noise):
            
                #pick up randomly noise file 
                filenoise=self.file_path_noise+fichiers_noise[randint(0, len(fichiers_noise)-1)]
         
                audio_amps, original_sample_rate = self.read_audio_file(filenoise)
                while (audio_amps.shape[0]<int(self.segment_duration*original_sample_rate)):
                    filenoise=self.file_path_noise+fichiers_noise[randint(0, len(fichiers_noise)-1)]
                    audio_amps, original_sample_rate = self.read_audio_file(filenoise)
            
                amplitudes_n, sample_rate = self.downsample_file(audio_amps, original_sample_rate, sample_rate)

                #pick up randomly time 
                time=randint(0, (amplitudes_n.shape[0]-int(self.segment_duration*sample_rate)))
                
                
                
                audio_noise=amplitudes_n[time:time+(int(self.segment_duration*sample_rate))]
                  
        
                augmented.append(self.blend(audio, audio_noise, 0.6, 0.4))
            

        return augmented
    
    def get_annotation_information(self, annotation_file_name, original_sample_rate):


            # Process the .svl xml file
            xmldoc = minidom.parse(self.folder_annotation+'/'+annotation_file_name+'.svl')
            itemlist = xmldoc.getElementsByTagName('point')
            idlist = xmldoc.getElementsByTagName('model')

            start_time = []
            end_time = []
            labels = []
            audio_file_name = ''

            if (len(itemlist) > 0):

                
                print (annotation_file_name)
                
                # Iterate over each annotation in the .svl file (annotatation file)
                for s in itemlist:

                    # Get the starting seconds from the annotation file. Must be an integer
                    # so that the correct frame from the waveform can be extracted
                    start_seconds = float(s.attributes['frame'].value)/original_sample_rate
                    
                    # Get the label from the annotation file
                    label = str(s.attributes['label'].value)

                    # Set the default confidence to 10 (i.e. high confidence that
                    # the label is correct). Annotations that do not have the idea
                    # of 'confidence' are teated like normal annotations and it is
                    # assumed that the annotation is correct (by the annotator). 
                    label_confidence = 10

                    # Check if a confidence has been assigned
                    if ',' in label:

                        try:     
                            # Extract the raw label
                            lalel_string = label[:label.find(','):]

                            # Extract confidence value
                            label_confidence = int(label[label.find(',')+1:])

                            # Set the label to the raw label
                            label = lalel_string
                        except : 
                            raise TypeError("the label confidence number is missing on this file") 

                    # If a file has a blank label then skip this annotation
                    # to avoid mislabelling data
                    if label == '':
                        break

                    # Only considered cases where the labels are very confident
                    # 10 = very confident, 5 = medium, 1 = unsure this is represented
                    # as "SPECIES:10", "SPECIES:5" when annotating.
                    if label_confidence > 1 :
                        # Get the duration from the annotation file
                        annotation_duration_seconds = float(s.attributes['duration'].value)/original_sample_rate
                        start_time.append(start_seconds)
                        end_time.append(start_seconds+annotation_duration_seconds)
                        labels.append(label)

            df_svl = pd.DataFrame({'Start': start_time, 'End':end_time ,'Label': labels})
            return df_svl 

    def get_meta_label(self, file_name_no_extension, database_ref, dictionary):
            xeno_id=file_name_no_extension.split('_')[5][2:]
            
            
            assert np.isin(xeno_id, database_ref['id']), "the id of the file {} is not contained in the reference database".format(file_name_no_extension)
                
            try: 
                meta=database_ref[database_ref['id']==int(xeno_id)][['lat','lng','date','time','cnt']].values[0]

                y=dictionary[database_ref[database_ref['id']==int(xeno_id)]['Scientifique_name'].values[0]]
            except : 
                raise TypeError("the metadata are not provided in the file") 
                
            return meta, y

    def getXY (self, audio_amplitudes, start_sec, annotation_duration_seconds, 
                   label, file_name_no_extension, database_ref, labelName_to_labelInd, verbose):
        
            X_segments = []
            X_meta_segments = []        
            Y_labels = []
            
            # Calculate how many segments can be extracted based on the duration of
            # the annotated duration. If the annotated duration is too short then
            # simply extract one segment. If the annotated duration is long enough
            # then multiple segments can be extracted.
            if annotation_duration_seconds-self.segment_duration < 0:
                segments_to_extract = 1
            else:
                segments_to_extract = annotation_duration_seconds-self.segment_duration+1
            
            if verbose:
                print ("segments_to_extract", segments_to_extract)


            for i in range (0, segments_to_extract):
                if verbose:
                    print ('Segment {} of {}'.format(i+1, segments_to_extract))
                    print ('*******************')
                
                # Set the correct location to start with.
                # The correct start is with respect to the location in time
                # in the audio file start+i*sample_rate
                start_data_observation = start_sec*self.downsample_rate+i*(self.downsample_rate)
                # The end location is based off the start
                end_data_observation = start_data_observation + (self.downsample_rate*self.segment_duration)
            
                # This case occurs when something is annotated towards the end of a file
                # and can result in a segment which is too short.
                if end_data_observation > len(audio_amplitudes):
                    continue
                
                # Extract the segment of audio
                X_audio = audio_amplitudes[start_data_observation:end_data_observation]

                # Determine the actual time for the event
                start_time_seconds = start_sec + i
                                
                
                # Extract some meta data (here, the hour at which the vocalisation event occured)
                meta_location, label=self.get_meta_label(file_name_no_extension, database_ref, labelName_to_labelInd)
                
                X_segments.append(X_audio)
                X_meta_segments.append(meta_location)
                Y_labels.append(label)
                
            return X_segments, X_meta_segments, Y_labels
            
            
            
    def create_dataset(self, verbose):   
            
            
            'for each folder and each file of bird song'
        
            #create folder containig the results
            if not os.path.exists(self.out_dir):
                    os.makedirs(self.out_dir)
            else:
                    print("outdir already created")

            labelName_to_labelInd=self.load_dico(self.out_dir+'/labelName_to_labelInd_22.json',key_int=False,print_me=False)                
            fichiers=[f for f in listdir(self.folder)] 
            database_ref=pd.read_csv(self.database_file,sep=';',encoding = "ISO-8859-1")  
                
            X_calls = []
            X_meta = []
            Y_calls = []    

            for file in fichiers: 

                file_name_no_extension=file[:-4]
                audio_amps, audio_sample_rate = self.read_audio_file(self.folder+'/'+file)
                filtered = self.butter_lowpass_filter(audio_amps, self.lowpass_cutoff, self.nyquist_rate)
                amplitudes, sample_rate = self.downsample_file(filtered, audio_sample_rate, self.downsample_rate)
                        
                annotation= self.get_annotation_information(file[:-4],audio_sample_rate)
                  
                for index, row in annotation.iterrows():

                    start_seconds = int(round(row['Start']))
                    end_seconds = int(round(row['End']))
                    label = row['Label']
                    annotation_duration_seconds = end_seconds - start_seconds

                    # Extract augmented audio segments and corresponding binary labels
                    X_data, X_meta_, Y = self.getXY(amplitudes, start_seconds, 
                                                            annotation_duration_seconds, label, 
                                                            file_name_no_extension, database_ref, labelName_to_labelInd,  verbose)
                            
                    X_calls.extend(X_data)
                    X_meta.extend(X_meta_)
                    Y_calls.extend(Y)
                            
            if self.type_saved_data=='image':
                X_calls = self.convert_all_to_image(X_calls, sample_rate)

                  
            X_calls, X_meta, Y_calls = np.asarray(X_calls), np.asarray(X_meta), np.asarray(Y_calls)
              
            return X_calls, X_meta, Y_calls
                       
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
            
                
    def load_data_from_pickle(self, path, X, X_meta, Y,):
            '''
            Load all of the spectrograms from a pickle file
        
            '''
            infile = open(os.path.join(path, X+'.pkl'),'rb')
            X = pickle.load(infile)
            infile.close()
        
            infile = open(os.path.join(path, X_meta+'.pkl'),'rb')
            X_meta = pickle.load(infile)
            infile.close()
        
            infile = open(os.path.join(path, Y+'.pkl'),'rb')
            Y  = pickle.load(infile)
            infile.close()
        
            return X, X_meta, Y  
