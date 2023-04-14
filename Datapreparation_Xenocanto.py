# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:44:09 2022

@author: ljeantet
"""

from Preprocessing_Xenocanto import *
import params 


folder='C:/Users/ljeantet/Documents/Postdoc/Location/Xenocanto/Audio_files/Training_1'
folder_annotation='C:/Users/ljeantet/Documents/Postdoc/Location/Xenocanto/Annotation/Training_1'
out_dir='C:/Users/ljeantet/Documents/Postdoc/Location/Xenocanto/out'

database_file='Xenocanto_metadata_qualityA_selection.csv'

out_dir="Xenocanto/out"


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



#####################################################################

#check for error     

labelName_to_labelInd=pre_pro.load_dico(out_dir+'/labelName_to_labelInd.json',key_int=False,print_me=False)                
fichiers=[f for f in listdir(folder)] 
file=fichiers[265]

'for each folder and each file of bird song'
  
      #create folder containig the results
      if not os.path.exists(out_dir):
          os.makedirs(self.file_out)
      else:
          print("outdir already created")


      #prepare database ref and load dict    

      #database_ref['date']=pd.to_datetime(database_ref['date'].astype(str),format='%d/%m/%Y')

      labelName_to_labelInd=pre_pro.load_dico(out_dir+'/labelName_to_labelInd.json',key_int=False,print_me=False)                
      fichiers=[f for f in listdir(folder)] 
      
      X_calls = []
      X_meta = []
      Y_calls = []    
      fichiers[45:53]
      file=fichiers[265]
      database_ref=pd.read_csv(database_file,sep=';',encoding = "ISO-8859-1")   
            for file in fichiers_wav: 

                
                  file_name_no_extension=file[:-4]
                  audio_amps, audio_sample_rate = pre_pro.read_audio_file(folder+'/'+file)
                  amplitudes, sample_rate = pre_pro.downsample_file(audio_amps, audio_sample_rate, pre_pro.downsample_rate)
                  
                  annotation= pre_pro.get_annotation_information(file[:-4],audio_sample_rate)
                  
                  for index, row in annotation.iterrows():
                      print(index)
                      start_seconds = int(round(row['Start']))
                      end_seconds = int(round(row['End']))
                      label = row['Label']
                      annotation_duration_seconds = end_seconds - start_seconds

                          # Extract augmented audio segments and corresponding binary labels
                      X_data, X_meta_, Y = pre_pro.getXY(amplitudes, start_seconds, 
                                                      annotation_duration_seconds, label, 
                                                      file_name_no_extension, database_ref, labelName_to_labelInd,  verbose)
                      
                      X_calls.extend(X_data)
                      X_meta.extend(X_meta_)
                      Y_calls.extend(Y)
                      
                      
            X_calls, X_meta, Y_calls = np.asarray(X_calls), np.asarray(X_meta), np.asarray(Y_calls)


Paridae_Saxicola_rubetra_Poland_2020-05-16_XC689996_sex .svl

(self, audio_amplitudes, start_sec, annotation_duration_seconds, 
               label, file_name_no_extension, database_ref, labelName_to_labelInd, verbose)

row=annotation.iloc[0,:]
start_sec=start_seconds
file_name_no_extension=file[:-4]
segment_duration=5
X_augmented_segments = []
X_meta_augmented_segments = []

Y_augmented_labels = []
    
# Calculate how many segments can be extracted based on the duration of
# the annotated duration. If the annotated duration is too short then
# simply extract one segment. If the annotated duration is long enough
# then multiple segments can be extracted.
if annotation_duration_seconds-segment_duration < 0:
    segments_to_extract = 1
else:
    segments_to_extract = annotation_duration_seconds-segment_duration+1
    
if verbose:
    print ("segments_to_extract", segments_to_extract)


for i in range (0, segments_to_extract):
    if verbose:
        print ('Semgnet {} of {}'.format(i, segments_to_extract-1))
        print ('*******************')
        
    # Set the correct location to start with.
    # The correct start is with respect to the location in time
    # in the audio file start+i*sample_rate
    start_data_observation = start_sec*sample_rate+i*(sample_rate)
    # The end location is based off the start
    end_data_observation = start_data_observation + (sample_rate*segment_duration)
    
    # This case occurs when something is annotated towards the end of a file
    # and can result in a segment which is too short.
    if end_data_observation > len(amplitudes):
        continue
    print('done')
    # Extract the segment of audio
    X_audio = amplitudes[start_data_observation:end_data_observation]

    # Determine the actual time for the event
    start_time_seconds = start_sec + i

    # Extract some meta data (here, the hour at which the vocalisation event occured)
    meta_location, label =pre_pro.get_meta_label(file_name_no_extension, database_ref, labelName_to_labelInd)

    X_augmented_segments.append(X_audio)
    X_meta_augmented_segments.append(meta_location)
    Y_augmented_labels.append(label)
  
    print('done 1')
    

    def get_meta_label(self, database_ref, file_name_no_extension, dictionary)
        xeno_id=file_name_no_extension.split('_')[5][2:]

        meta=database_ref[database_ref['id']==int(xeno_id)][['lat','lng','date','time','cnt']].values

        y=dictionary[database_ref[database_ref['id']==int(xeno_id)]['Scientifique_name'].values[0]]

        return meta, y
    
    
    
    
annotation_file_name=file[:-4]
original_sample_rate=audio_sample_rate
    
def get_annotation_information(self, annotation_file_name, original_sample_rate):


        # Process the .svl xml file
        xmldoc = minidom.parse(folder_annotation+'/'+annotation_file_name+'.svl')
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
                print(s)
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


########plot result

image=convert_single_to_image(audio_amps,audio_sample_rate,'mel-spectro' )
pre_pro.print_spectro(image, audio_sample_rate, title="spectogram")


audio_amps, audio_sample_rate = pre_pro.read_audio_file(folder+'/'+file)
filtered = pre_pro.butter_lowpass_filter(audio_amps, lowpass_cutoff, nyquist_rate)

amplitudes, sample_rate = pre_pro.downsample_file(filtered, audio_sample_rate, pre_pro.downsample_rate)


image=pre_pro.convert_single_to_image(X_calls[0],audio_sample_rate)
pre_pro.print_spectro(image, audio_sample_rate, title="spectogram")

image.shape

image_1=convert_single_to_image(audio_amps,audio_sample_rate,'mel-spectro' )
pre_pro.print_spectro(image_1, audio_sample_rate, title="spectogram")


image_2=convert_single_to_image(amplitudes,audio_sample_rate,'mel-spectro' )
pre_pro.print_spectro(image_2, audio_sample_rate, title="spectogram")


S2=X_calls
S2.shape
librosa.display.specshow(S2[627,:,:],sr=22050,hop_length=32,cmap='magma')



