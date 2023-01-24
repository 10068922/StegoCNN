############################################################################
##                    Main Module for Model Training                      ##
############################################################################

# Import libraries
import os
import glob
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

from datetime import datetime
from time import time

from keras import callbacks
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop, Adam
from tensorflow_addons.optimizers import AdamW
from keras.utils import plot_model

from data_manager import DataManager
from model import CNNModel
from plot_data import PlotData

K.image_data_format()
os.environ["PATH"] += os.pathsep + r'C:/Program Files/Graphviz/bin'

# Image dataset
image_dataset = 'alaska2'
#image_dataset = 'IStego100K'

# Model type
model_type = 'S-Model'
#model_type = 'TL-Model'
#model_type = 'EN-Model'

# test dataset img
#model_dataset = 'dataset_' + model_type

# Images width, height & channels
img_height = 512
img_width = 512
num_channels = 3

# Input image shape
image_shape = (img_height, img_width, num_channels)

# Class Number
class_number = 2

# Loss function
model_loss_function = 'binary_crossentropy'

# Model metrics to evaluate training
model_metrics = ["accuracy"]

# Dataset paths
image_base_dir = 'D:/CNN/Dataset/'
train_data_dir = image_base_dir + 'train/' + image_dataset
validation_data_dir = image_base_dir + 'validation/' + image_dataset
train_data_dir = image_base_dir + 'train'
validation_data_dir = image_base_dir + 'validation'

# Output paths
base_path = 'D:/CNN/StegoCNN'
train_path = base_path + '/trained/' + model_type
model_png = train_path + '/model.png'
model_summary_file = train_path + '/model_summary.txt'
model_arch_file = train_path + '/model.json'
model_classid_file = train_path + '/model_classid.json'
model_loss_acc = train_path + '/model_loss_acc.png'
log_path = train_path + '/log'
train_log_file = log_path + '/model_train.csv'
train_checkpoint_loss_file = log_path + '/Best-weights-loss-{epoch:03d}-{loss:.4f}.h5'
train_checkpoint_accuracy_file = log_path + '/Best-weights-accuracy-{epoch:03d}-{val_accuracy:.4f}.h5'
model_training_log = base_path + '/training_log/'
model_train_time = train_path + '/model_train_time.txt'

# Model training parameters
num_of_epoch = 2
num_of_train_samples = 56000
num_of_validation_samples = 24000
num_of_train_samples = 400
num_of_validation_samples = 80

# Batch size
train_batch_size = 32
val_batch_size = 32

# Define optimizers
# learning rate
lr=0.001
# momentum
m=0.8
model_optimizer_sgd = SGD(learning_rate=lr, decay=1e-6, momentum=m, nesterov=True)
model_optimizer_rmsprop = RMSprop(learning_rate=lr)
model_optimizer_adam = Adam(learning_rate=lr, decay=0.00001)
model_optimizer_adamW = AdamW(learning_rate=lr, weight_decay=0.004)
# For testing of different optimizers
model_optimizer = model_optimizer_sgd


# Delete a file
def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        pass


# Save model summary
def save_summary(s):
    with open(model_summary_file, 'a') as f:
        f.write('\n' + s)
        f.close()
        pass


# define learning rate schedule
def build_lrfn():
    # Detect hardware for returning appropriate distribution strategy
    try:
        # TPU detection
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and GPU.
        strategy = tf.distribute.get_strategy()

    lr_start = 0.001

    lr_max = 0.008 * strategy.num_replicas_in_sync
    lr_min = 0.001 
    lr_rampup_epochs = 1
    lr_sustain_epochs = 2
    lr_exp_decay = .8

    def lrfn(epoch) :
        if epoch < lr_rampup_epochs :
            lr = lr_start + (lr_max-lr_min) / lr_rampup_epochs * epoch
        elif epoch < lr_rampup_epochs + lr_sustain_epochs :
            lr = lr_max
        else :
            lr = lr_min + (lr_max - lr_min) * lr_exp_decay**(epoch - lr_sustain_epochs - lr_rampup_epochs)
        return lr
    
    return lrfn

# for plotting graphs of loss and accuracy
def train_valid_plot(train_data, start_epoch):
    #Plot the training and validation data
    train_acc=train_data.history['accuracy']
    #train_acc=train_data.history['acc']
    train_loss=train_data.history['loss']
    valid_acc=train_data.history['val_accuracy']
    #valid_acc=train_data.history['val_acc']
    valid_loss=train_data.history['val_loss']
    Epoch_count=len(train_acc)+ start_epoch
    Epochs=[]
    for i in range (start_epoch,Epoch_count):
        Epochs.append(i+1)   
    index_loss=np.argmin(valid_loss)#  this is the epoch with the lowest validation loss
    val_lowest=valid_loss[index_loss]
    index_acc=np.argmax(valid_acc)
    acc_highest=valid_acc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss + 1 + start_epoch)
    vc_label='best epoch= '+ str(index_acc + 1 + start_epoch)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8))
    axes[0].plot(Epochs,train_loss,'r', label='Training loss')
    axes[0].plot(Epochs,valid_loss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1+start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,train_acc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,valid_acc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1+start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    plt.show()
    plt.savefig(model_loss_acc)
    
def main():
    # Init the DataManager class
    print("========================== Load dataset ============================")
    dataManager = DataManager(img_height, img_width)
    
    # Get data
    print("Training data directory => ", train_data_dir)
    print("Validation data directory =>", validation_data_dir)
    print("Training batch size =>", train_batch_size)
    print("Validation batch size =>", val_batch_size)
    print("")
    train_data, validation_data = dataManager.get_train_data(
        train_data_dir, validation_data_dir, train_batch_size, val_batch_size)    
    
    # Get class name:id
    label_map = (train_data.class_indices)
    
    # Save model class id
    isExist = os.path.exists(train_path)
    if not isExist:
       os.makedirs(train_path)
   
    with open(model_classid_file, 'w') as outfile:
        json.dump(label_map, outfile)
    
    # Init the model    
    cnnModel = CNNModel(image_shape, class_number)
    print("")
    print("Image shape => ", image_shape)
    print("Class number => ", class_number)
    print("")
    
    # Get model architecture
    print("====================== Load model architecture =====================")
    model = cnnModel.get_model_architecture()
    # plot the model
    plot_model(model, to_file=model_png, show_shapes=True, show_layer_names=True)
    
    # serialize model to JSON
    model_json = model.to_json()
    
    print("model_arch_file =>", model_arch_file)
    print("")
    with open(model_arch_file, "w") as json_file:
        json_file.write(model_json)

    model = cnnModel.compile_model(model, model_loss_function, model_optimizer, model_metrics)

    # Save model summary
    model.summary(print_fn=save_summary)

    # Prepare callbacks
    lrfn = build_lrfn()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')

    csv_log = callbacks.CSVLogger(train_log_file, separator=',', append=False)
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    checkpoint_loss = callbacks.ModelCheckpoint(train_checkpoint_loss_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint_acc = callbacks.ModelCheckpoint(train_checkpoint_accuracy_file, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir = model_training_log + "{}".format(time()))
    # for fixed learning rate
    #callbacks_list = [csv_log, tensorboard, checkpoint_loss, checkpoint_acc]
    # for learning rate schedule
    callbacks_list = [csv_log, tensorboard, checkpoint_loss, checkpoint_acc, reduce_lr_loss, lr_schedule]    

    print("===================== Start training model =========================")
    start_time = datetime.now()
    sdt_string = start_time.strftime("%d/%m/%Y %H:%M:%S")
    print("start training date and time => ", sdt_string)
    # start training

    history = model.fit(train_data,
                        steps_per_epoch=num_of_train_samples // train_batch_size,
                        epochs=num_of_epoch,
                        validation_data=validation_data,
                        validation_steps=num_of_validation_samples // val_batch_size,
                        verbose=1,
                        callbacks=callbacks_list)
   
    end_time = datetime.now()
    edt_string = end_time.strftime("%d/%m/%Y %H:%M:%S")
    print("end training date and time => ", edt_string)
    t = "total training time => " + str((end_time - start_time).total_seconds() / 60.0) + " mins"
    print(t)
    with open(model_train_time, 'a') as f:
            f.write('\nStart time: ' + sdt_string)
            f.write('\nEnd time: ' + edt_string)
            f.write('\n' + t)
            f.close()    
        
    print("=================== training process completed =====================")

    # Plot the chart for model training history
    train_valid_plot(history,0)

    # Plot the chart for accuracy and loss on both training and validation
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs,acc,'r',label='Training Accuracy')
    plt.plot(epochs,val_acc,'b',label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs,loss,'r',label='Training Loss')
    plt.plot(epochs,val_loss,'b',label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
