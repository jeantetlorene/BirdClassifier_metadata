# Empowering Deep Learning Acoustic Classifiers with Human-like Ability to Utilize Contextual Information for Wildlife Monitoring

## Introduction

The goal of this project is to enhance deep learning classifiers in bioacoustics by incorporating contextual meta-data, such as time and location, just as humans utilize contextual information to identify species calls from acoustic recordings. In this study, we explore different methods to provide location information to a bird song classifier.

The data are available on Zenodo :  10.5281/zenodo.7828148 

## Authors 
- Lorène Jeantet  
- Emmanuel Dufourq

African Institute for Mathematical Sciences, Cape Town, South Africa

Machine Learning for Ecology research group

## Data Availability 

The dataset used in this project can be downloaded from Zenodo at the following link: [Zenodo Dataset Link](https://doi.org/10.5281/zenodo.7828148).


## Data Description

The audio files are provided in the "Audio.zip" archive, and the manually verified annotations can be found in "Annotations.zip". Each file follows the following naming convention: `Family_genus_species_country of recording_date of recording_Xeno-canto ID number_type of song.wav/svl`. Additional metadata information for each file can be found in the provided CSV file, "Xenocanto_metadata_qualityA_selection", which is based on the Xeno-canto ID number.

To view the annotations, we recommend using the Sonic Visualiser software.

The dataset has been divided into two folders: "Training" and "Validation". This split was done to train and evaluate the efficiency of each method. For each species and country, 70% of the downloaded recordings were randomly selected for the training dataset, while the remaining 30% were kept for validation. 

The data necessary to trained the Geographical prior can be found in the folder Geographical_prior/Data. It consists of pre-processed csv file obtained from Xenocanto containing the meta-data information of all the recordings of quality A (Xenocanto_metadata_allrecordings_qualityA.csv) and quality B (Xenocanto_metadata_allrecordings_qualityB.csv).  

To ensure proper execution of the provided Python scripts, we recommend keeping the data organized in the aforementioned folder structure.
Please adjust the headings and sections according to your specific project and repository structure as needed.

## Libraries

Tested on Python 3.11
- soundfile==0.12.1
- librosa==0.10.1
- numpy==1.23.5
- yattag==1.15.1
- pandas==2.0.1
- scipy==1.10.1
- scikit-learn==1.2.2
- matplotlib==3.7.1
- tensorflow==2.12.0

## Code description 

### Organisation of the folders

The main folder contains the scripts necessary to pre-process the dataset and to train the Baseline model as well as the multi-branch CNN. The scripts to train the Geographical prior can be found in the dedicated folder "Geographical_prior". 

The scripts are written to save the pre-processed data into the "ou"' folder and the trained models, along with their outputs, into the "Models_out" folder. When training a model, a new folder associated with that experiment will be created in "Models_out" with the date of the experiment and the model name as the folder's name.

The "out" folder contains dictionaries used in the article to map between an index number and the name of the species or the country. Functions to open the dictionnaries or generate them can be found in the "Preprocessing_Xenocanto" class ("Preprocessing_Xenocanto.py"). 

For the Greographical prior, the data necessary to train the model can be found in the "Data" folder. Similarly to the main folder, when training the model, a new folder associated with that experiment will be created in 'Models_out' with the date of the experiment and the model name as the folder's name.

### Data pre-processing

### Training of the Baseline Model and Multi-branch CNN 

### Training and Application of the Geographical Prior 

THe codes related to the training process of the Geographical prior can be found in the folder Geographical_prior. 