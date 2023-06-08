# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:19:31 2022

@author: ljeantet
"""

import os
os.chdir('C:/Users/ljeantet/Documents/Postdoc/Location/Presence_Only_geo_prior/Xenocanto')

from Prior_geo_Xenocanto_Preprocessing import *
from FCNet_model import *
import losses

import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'use_photographers', default=False,
    help=('Include photographers classifier to the model'))

#1- Import dataset
folder='Data/'   #folder where we can find the dataset
database='Xenocanto_metadata_allrecordings_qualityA.csv'  #exact name of dataset
out_dir='out' #name of the out folder
validation='Xenocanto_metadata_allrecordings_qualityB_cleaned.csv'
num_classes=8994

loc_encode='encode_cos_sin'
embed_dim=256
use_batch_normalization=False
num_inputs=4
lr=0.0005 #Initial learning rate
lr_decay=0.98
epochs=10
random_seed=42
batch_size=32
max_instances_per_class=50

def build_input_data(folder, database, out_dir, loc_encode, batch_size, is_training, num_classes=None):
    
    pre_pro=Prior_geo_MetaData_Generator(folder, database, 
                                         out_dir, 
                                         loc_encode, 
                                         batch_size, 
                                         max_instances_per_class=(max_instances_per_class if is_training \
                                                                                                else -1),
                                         is_training=is_training,
                                         num_classes=num_classes)


    return pre_pro.make_meta_dataset()
    
   

def lr_scheduler(epoch, lr):
  if epoch < 1:
      return lr
  else:
      return lr * lr_decay


def train_model(model,
                dataset,
                num_train_instances,
                val_dataset,
                loss_o_loc,
                loc_p_loss,
                p_o_loss):
  
    
    
    
    #model_dir=create_new_folder('Models_out/', 'FCNet')
    #summary_dir = os.path.join(model_dir, "summaries")
    #summary_callback = tf.keras.callbacks.TensorBoard(summary_dir,
                                                    profile_batch=0)
    #checkpoint_filepath = os.path.join(model_dir, "ckp")
    #checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_freq='epoch')
    
    #to save the results
    seed=random.randint(1, 1000000)
    dir_out=create_new_folder('Models_out/', 'FCNet')  #attention :supprime donnees enregistrees si deja existant
    filepath= dir_out+"/weights_{}.hdf5".format(seed)
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1, save_best_only=True, mode='max')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,verbose=1, save_weights_only=True,save_freq=2*batch_size)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    #callbacks = [summary_callback, checkpoint_callback, lr_callback]

    model.compile(optimizer=optimizer,loss=loss_o_loc)

    return model.fit(dataset,
                   epochs=epochs,
                   steps_per_epoch=int(num_train_instances/batch_size),
                   callbacks=[cp_callback],
                   validation_data=val_dataset)

def set_random_seeds():
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

def main(_):
    
    global num_classes
    set_random_seeds()
    
    dataset, num_instances, num_classes = build_input_data(folder, database, out_dir, loc_encode, batch_size, is_training=True, num_classes=num_classes)
    
    randgen = RandSpatioTemporalGenerator(
      loc_encode=loc_encode)

    val_dataset, _, _ = build_input_data(folder, validation, out_dir, loc_encode, batch_size, is_training=False, num_classes=num_classes)

    model = FCNet(num_inputs=4,
                embed_dim=embed_dim,
                num_classes=num_classes,
                rand_sample_generator=randgen,
                use_bn=use_batch_normalization)

    loss_o_loc = losses.weighted_binary_cross_entropy(pos_weight=num_classes)
    loc_p_loss = losses.log_loss()
    p_o_loss = losses.weighted_binary_cross_entropy(pos_weight=num_classes)

    model.build((None, num_inputs))
    model.summary()

    #train_model(model, dataset, num_instances, val_dataset, loss_o_loc, loc_p_loss, p_o_loss)

if __name__ == '__main__':
    app.run(main)






