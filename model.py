############################################################################
##                       Models for Image Steganalysis                    ##
############################################################################


# Import libraries
import tensorflow as tf
import tensorflow.keras.layers as L

#from efficientnet_pytorch import EfficientNet

from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import ELU, PReLU, LeakyReLU
from keras.layers import Concatenate, Lambda, Layer, InputLayer, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import DenseNet121, MobileNet, MobileNetV2, ResNet50V2
from keras.applications.nasnet import NASNetMobile
from tensorflow.python.keras.models import Input

K.image_data_format()

class CNNModel(object):

    def __init__(self, input_shape, nbr_classes):
        super(CNNModel, self).__init__()
        self.nbr_classes = nbr_classes
        self.input_shape = input_shape

    # Defining the model
    def get_model_architecture(self):
        #Scratch model (S-Model)
        # model = Sequential()

        # # Input Layer
        # model.add(InputLayer(input_shape=self.input_shape))

        # # Hidden Layer
        # model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=2,
        #                  padding="same", use_bias=False, trainable=False))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=.1))


        # model.add(Conv2D(filters=5, kernel_size=(5, 5), strides=2,
        #                  padding="same", use_bias=True, trainable=True))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=.1))
        # model.add(Dropout(0.1))
        
     
        # model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=2,
        #                  padding="same", use_bias=True, trainable=True))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=.1))
        # model.add(Dropout(0.1)) 
     

        # Classification
        # Converts 2D feature maps to 1D feature vectors
        # model.add(Flatten())
        # model.add(Dense(200))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=.1))
        # model.add(Dense(200))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=.1))
        # model.add(Dropout(0.5))

        # model.add(Dense(2))
        # model.add(Activation('softmax'))

        #Transfer learning model (TL-Model)
        #base_model = DenseNet121(
        #base_model = MobileNetV2(
        #base_model = NASNetMobile(
        base_model = ResNet50V2(
        input_shape=self.input_shape,
        include_top=False)  # Do not include the ImageNet classifier at the top.
        base_model.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(2, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        #Ensemble model (EN-Model)
        # base_model1 = keras.models.load_model('Best-weights-accuracy-DenseNet.h5', compile=False) 
        # base_model1._name = 'model1'
        # base_model2 = keras.models.load_model('Best-weights-accuracy-MobileNet.h5', compile=False) 
        # base_model2._name = 'model2'
        # base_model3 = keras.models.load_model('Best-weights-accuracy-SModel.h5', compile=False) 
        # base_model3._name = 'model3'
        # models = [base_model1, base_model2, base_model3] #stacking individual models
        # model_input = tf.keras.Input(shape=self.input_shape)
        # model_outputs = [model(model_input) for model in models] #collect outputs from the models
        # ensemble_output = tf.keras.layers.Average()(model_outputs) #averaging outputs
        # model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)
        
        model.summary()

        return model

    def compile_model(self, model, loss, optimizer, metrics):
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model
