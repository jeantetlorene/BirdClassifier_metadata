# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:44:09 2022

@author: ljeantet
"""

'''
Script related to the article ...
Script to process the audio files downloaded from Xenocanto
'''


from Preprocessing_Xenocanto import *
import params 

#path for the different folders containing audio and annotation files
folder='C:/Users/ljeantet/Documents/Postdoc/Location/Xenocanto/Audio_files/Training_1'
folder_annotation='C:/Users/ljeantet/Documents/Postdoc/Location/Xenocanto/Annotation/Training_1'
#file obtained from Xenocanto containing the metadata of the recordings of the species selected for the study 
database_file='Xenocanto_metadata_qualityA_selection.csv'


#folder where the processed data will be saved
out_dir='C:/Users/ljeantet/Documents/Postdoc/Location/Xenocanto/out'



#parameters get back from the params.py file
n_fft = params.n_fft # Hann window length
hop_length = params.hop_length
n_mels=params.n_mels
nyquist_rate =params.nyquist_rate
lowpass_cutoff =params.lowpass_cutoff
win_length=params.win_length
f_min = params.f_min # Spectrogram, minimum frequency for call
f_max=params.f_max



segment_duration=params.segment_duration
hop_chunck=params.hop_chunck
nb_augmented_noise=params.nb_augmented_noise
nb_augmented_pitch=params.nb_augmented_pitch

type_spec=params.type_spec
downsample_rate = params.downsample_rate


type_saved_data='audio'

verbose=False


#training dataset
pre_pro = Preprocessing_Xenocanto(folder,folder_annotation, out_dir, database_file, lowpass_cutoff, 
             downsample_rate,  segment_duration, hop_chunck, nb_augmented_noise,  
             nb_augmented_pitch, type_spec, type_saved_data,                
             n_fft, hop_length, n_mels, f_min, f_max, nyquist_rate, win_length)


X_calls, X_meta, Y_calls = pre_pro.create_dataset(verbose)

pre_pro.save_data_to_pickle(X_calls, X_meta, Y_calls, Saved_X='X_Xenocanto_audio_training_1-pow', Saved_meta='X_meta_Xenocanto_audio_training_1-pow',Saved_Y='Y_Xenocanto_audio_training_1-pow')

#validation dataset 
folder='C:/Users/ljeantet/Documents/Postdoc/Location/Xenocanto/Audio_files/Validation_1'
folder_annotation='C:/Users/ljeantet/Documents/Postdoc/Location/Xenocanto/Annotation/Validation_1'
out_dir='C:/Users/ljeantet/Documents/Postdoc/Location/Xenocanto/out'



type_saved_data='image'
pre_pro = Preprocessing_Xenocanto(folder,folder_annotation, out_dir, database_file, lowpass_cutoff, 
             downsample_rate,  segment_duration, hop_chunck, nb_augmented_noise,  
             nb_augmented_pitch, type_spec, type_saved_data,                
             n_fft, hop_length, n_mels, f_min, f_max, nyquist_rate, win_length)


X_calls_val, X_meta_val, Y_calls_val = pre_pro.create_dataset(verbose)

pre_pro.save_data_to_pickle(X_calls_val, X_meta_val, Y_calls_val, Saved_X='X_Xenocanto_melspect_val_1-pow', Saved_meta='X_meta_Xenocanto_melspect_val_1-pow',Saved_Y='Y_Xenocanto_melspect_val_1-pow')


##################################################################
#Explore Y 



X_calls, X_meta, Y_calls = pre_pro.load_data_from_pickle(out_dir, 'X_Xenocanto-pow', 'X_meta_Xenocanto-pow', 'Y_Xenocanto-pow')



X_calls, X_meta, Y_calls = pre_pro.load_data_from_pickle(out_dir, 'X_Xenocanto-pow', 'X_meta_Xenocanto-pow', 'Y_Xenocanto-pow')
X_calls_val, X_meta_val, Y_calls_val = pre_pro.load_data_from_pickle(out_dir, 'X_Xenocanto_audio_fmin150_fmax15000_validation_1-pow', 'X_Xenocanto_audio_fmin150_fmax15000_validation_1-pow', 'Y_Xenocanto_audio_fmin150_fmax15000_validation_1-pow')

Y=np.concatenate((Y_calls, Y_calls_val), axis=0)
Meta=np.concatenate((X_meta, X_meta_val), axis=0)


labelInd_to_labelName=pre_pro.load_dico(out_dir+'/labelInd_to_labelName_22.json',key_int=False,print_me=False)

Y_label=[labelInd_to_labelName[x.astype(str)] for x in Y]

countries=[x[4] for x in Meta]
df=pd.DataFrame({'label': Y_label, 'country':countries})


#plot histo country according species 
import seaborn as sns
sns.catplot(y="country",  hue="label", kind="count", data=df, palette="deep")

sns.color_palette("flare", as_cmap=True)
sns.color_palette()

species=np.unique(Y_label, return_counts=True)
species=np.unique(Y_label, return_counts=False)
country=np.unique(countries)

data=df.groupby(['label'])['country'].count()
df1.groupby(['State'])['Sales']

np.unique(Y_calls.astype(str), return_counts=True)


values_nb=[]
for specy in species :
    data=df[df['label']==specy]
    values_nb.extend(data.groupby(['country']).count().values)

np.mean(values_nb)
np.std(values_nb)
np.max(values_nb)
np.min(values_nb)
df.drop_duplicates()



#



