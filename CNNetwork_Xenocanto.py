# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:26:54 2022

@author: ljeantet
"""



'''
Script related to the article "Empowering Deep Learning Acoustic Classifiers with Human-like Ability 
to Utilize Contextual Information for Wildlife Monitoring" by Jeantet and Dufourq.

This class contains various architectures that we want to experiment with.

'''





from tensorflow.keras import backend, Input, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D, Concatenate, Embedding
from tensorflow.keras.models import Sequential
from tensorflow import keras


class CNNetwork:

    def __init__(self, X_shape_1, X_shape_2, conv_layers, conv_filters, dropout_rate, 
               conv_kernel,max_pooling_size,fc_units_1,fc_units_2,epochs,batch_size):
        self.conv_layers = conv_layers
        self.conv_filters = conv_filters
        self.dropout_rate = dropout_rate
        self.conv_kernel = conv_kernel
        self.max_pooling_size = max_pooling_size
        self.fc_units_1=fc_units_1
        self.fc_units_2=fc_units_2
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_shape_1=X_shape_1
        self.X_shape_2=X_shape_2

        
    def other_network(self):
        ''' Add other networks to this Class. '''
        
    def CNN_network(self):
        ''' Base line model (Case I in the article) '''
    
        cnn_input=Input(shape = (self.X_shape_1,self.X_shape_2,1), name="audio_input")
        feature_extractor = Conv2D(filters = self.conv_filters, kernel_size = self.conv_kernel, activation = 'relu')(cnn_input)
        feature_extractor = Dropout(rate = self.dropout_rate)(feature_extractor)
        feature_extractor = MaxPool2D(pool_size = self.max_pooling_size)(feature_extractor)
        
        for i in range(self.conv_layers):
            feature_extractor = Conv2D(filters = self.conv_filters, kernel_size = self.conv_kernel, activation = 'relu')(feature_extractor)
            feature_extractor = Dropout(rate = self.dropout_rate)(feature_extractor)
            feature_extractor = MaxPool2D(pool_size=self.max_pooling_size)(feature_extractor)
        
        feature_extractor = Flatten()(feature_extractor)
        
        #dense layer 1
        meta_ann = Dense(units = self.fc_units_1, activation='relu')(feature_extractor)
        meta_ann = Dropout(rate=self.dropout_rate)(meta_ann)
        
        
        #dense layer 2
        #meta_ann = Dense(units = self.fc_units_2, activation='relu')(meta_ann)
        #meta_ann = Dropout(rate=self.dropout_rate)(meta_ann)
        
  
        softmax_output=Dense(22, activation = 'softmax')(meta_ann)
        
        
        model = Model(inputs=cnn_input, outputs=softmax_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
               
        return model

    def custom_CNN_network(self):
        """ 
        Multi branch CNN taking spectrogram as input 1 and the name of the country as input 2 (case III in the article)
        
        
        """
        
        # Branch 1
        cnn_input=Input(shape = (self.X_shape_1,self.X_shape_2,1), name="audio_input")
        feature_extractor = Conv2D(filters = self.conv_filters, kernel_size = self.conv_kernel, activation = 'relu')(cnn_input)
        feature_extractor = Dropout(rate = self.dropout_rate)(feature_extractor)
        feature_extractor = MaxPool2D(pool_size = self.max_pooling_size)(feature_extractor)
        
        for i in range(self.conv_layers):
            feature_extractor = Conv2D(filters = self.conv_filters, kernel_size = self.conv_kernel, activation = 'relu')(feature_extractor)
            feature_extractor = Dropout(rate = self.dropout_rate)(feature_extractor)
            feature_extractor = MaxPool2D(pool_size=self.max_pooling_size)(feature_extractor)
        
        x1 = Flatten()(feature_extractor)
    
        
        # Branch 2
        meta_input = Input(shape=(2,1), name="meta_input")
        # Embedding the meta data 
        x=Embedding(50, 8, input_length=2)(meta_input)
        x2=Flatten()(x)
        
        # Concatenate
        combined = Concatenate()([x1, x2]) 
        
        
        #dense layer 1
        meta_ann = Dense(units = self.fc_units_1, activation='relu')(combined)
        meta_ann = Dropout(rate=self.dropout_rate)(meta_ann)
        
        #dense layer 2
        #meta_ann = Dense(units = self.fc_units_2, activation='relu')(meta_ann)
        #meta_ann = Dropout(rate=self.dropout_rate)(meta_ann)
        
        # Model softmax output
        softmax_output=Dense(22, activation = 'softmax', name="class_output")(meta_ann)
        
        model = Model(inputs=[cnn_input,meta_input], outputs=softmax_output)
        model.compile(loss={"class_output": keras.losses.CategoricalCrossentropy()}, optimizer='adam',
                      metrics={"class_output": ['accuracy']})
     

        
        
        return model
    
    def custom_CNN_network_1(self, X_meta_shape):
        """ 
        Multi branch CNN taking spectrogram as input 1 and the name of the country as input 2 coded in one-hot encoded vector (case II in the article)
        
        
        """
        
        
        
        
        # Branch 1
        cnn_input=Input(shape = (self.X_shape_1,self.X_shape_2,1), name="audio_input")
        feature_extractor = Conv2D(filters = self.conv_filters, kernel_size = self.conv_kernel, activation = 'relu')(cnn_input)
        feature_extractor = Dropout(rate = self.dropout_rate)(feature_extractor)
        feature_extractor = MaxPool2D(pool_size = self.max_pooling_size)(feature_extractor)
        
        for i in range(self.conv_layers):
            feature_extractor = Conv2D(filters = self.conv_filters, kernel_size = self.conv_kernel, activation = 'relu')(feature_extractor)
            feature_extractor = Dropout(rate = self.dropout_rate)(feature_extractor)
            feature_extractor = MaxPool2D(pool_size=self.max_pooling_size)(feature_extractor)
        
        x1 = Flatten()(feature_extractor)
    
        
        # Branch 2
        meta_input = Input(shape=(X_meta_shape), name="meta_input")
        
        
        # Concatenate
        combined = Concatenate()([x1, meta_input]) 
        
        
        #dense layer 1
        meta_ann = Dense(units = self.fc_units_1, activation='relu')(combined)
        meta_ann = Dropout(rate=self.dropout_rate)(meta_ann)
        
        #dense layer 2
        #meta_ann = Dense(units = self.fc_units_2, activation='relu')(meta_ann)
        #meta_ann = Dropout(rate=self.dropout_rate)(meta_ann)
        
        # Model softmax output
        softmax_output=Dense(22, activation = 'softmax', name="class_output")(meta_ann)
        
        model = Model(inputs=[cnn_input,meta_input], outputs=softmax_output)
        model.compile(loss={"class_output": keras.losses.CategoricalCrossentropy()}, optimizer='adam',
                      metrics={"class_output": ['accuracy']})
     

        
        
        return model
    

        


