############################################################################
##                        Evaluate a trained model                        ##
############################################################################

# import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error, classification_report, accuracy_score
from data_manager import DataManager
from plot_data import PlotData
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop, Adam
from tensorflow_addons.optimizers import AdamW
from keras import backend as K
K.image_data_format()

# Model type
model_type = 'S-Model'
#model_type = 'TL-Model'
#model_type = 'EN-Model'

# Images width, height & channels
img_height = 512
img_width = 512
num_channels = 3

# for plotting confusion matrix
cm_plot_labels = ['cover', 'stego']

# test dataset
model_test_dataset = 'dataset_' + model_type

# path to saved model files
base_path = 'D:/CNN/StegoCNN'
train_path = base_path + '/trained/' + model_type
saved_model_weights_path = train_path + '/Best-weights.h5'
saved_model_arch_path = train_path + '/model.json'
test_data_dir = 'D:/CNN/dataset/test'

# paths to save outputs
train_log_data_file = train_path + '/model_train.csv'
plt_cm = train_path + '/model_cm.png'
plt_normalized_cm = train_path + '/model_norm_cm.png'
plt_roc = train_path + '/model_roc.png'
plt_accuracy = train_path + '/model_accuracy.png'
plt_loss = train_path + '/model_loss.png'
eval_report = train_path + '/eval_report.txt'

# Loss function
model_loss_function = 'binary_crossentropy'


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

# model metrics to evaluate training
model_metrics = ["accuracy"]

# batch size
batch_size = 16

# Load architecture and weights of the model
def load_model():
    print("saved_model_arch_path: " + saved_model_arch_path)
    print("saved_model_weights_path:" + saved_model_weights_path)
    json_file = open(saved_model_arch_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(saved_model_weights_path)
    return loaded_model


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        pass


def main():
    # init DataManager class
    dataManager = DataManager(img_height, img_width)
    # init PlotData class
    plotData = PlotData()
    # load model
    model = load_model()
    # get test data
    test_data = dataManager.get_test_data(test_data_dir)
    # start the evaluation process
    print("===================== Start evaluaton =========================")
    y_true = test_data.classes
    # Confution Matrix and Classification Report
    Y_pred = model.predict(test_data)
    y_pred = np.argmax(Y_pred, axis=1)
    print(y_true)
    print(y_pred)
    # plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plotData.plot_confusion_matrix(cm, cm_plot_labels, plt_cm, title='Confusion Matrix')
    plotData.plot_confusion_matrix(cm, cm_plot_labels, plt_normalized_cm, normalize=True, title='normalized Confusion Matrix')
    # Compute ROC curve and ROC area for each class
    roc_auc = plotData.plot_roc(y_true, y_pred, plt_roc)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print('Mean Absolute Error (MAE): ' + str(mae))
    print('Mean Squared Error (MSE): ' + str(mse))
    print('Area Under the Curve (AUC): ' + str(roc_auc))
    c_report = classification_report(y_true, y_pred, target_names=cm_plot_labels)
    print(c_report)

    delete_file(eval_report)
    #Save the evaluation report
    with open(eval_report, 'a') as f:
        f.write('\n\n')
        f.write('******************************************************\n')
        f.write('**                 Evaluation Report                **\n')
        f.write('******************************************************\n')
        f.write('\n\n')
        f.write('Accuracy Score: ' + str(accuracy))
        f.write('\n\n')
        f.write('Mean Absolute Error (MAE): ' + str(mae))
        f.write('\n\n')
        f.write('Mean Squared Error (MSE): ' + str(mse))
        f.write('\n\n')
        f.write('Area Under the Curve (AUC): ' + str(roc_auc))
        f.write('\n\n')
        f.write('- Confusion Matrix:\n')
        f.write(str(cm))
        f.write('\n\n')
        f.write('Normalized Confusion Matrix:\n')
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        f.write(str(cm))
        f.write('\n\n')
        f.write('Classification report:\n')
        f.write(str(c_report))
        f.close()

    train_validation = ['train', 'validation']
    data = pd.read_csv(train_log_data_file)
    acc = data['accuracy'].values
    val_acc = data['val_accuracy'].values
    loss = data['loss'].values
    val_loss = data['val_loss'].values

     
    # plot graphs for loss and accuracy
    plotData.plot_2d(acc, val_acc, 'epoch', 'accuracy',
                     'Model Accuracy', train_validation, plt_accuracy)
    plotData.plot_2d(loss, val_loss, 'epoch', 'loss',
                     'Model Loss', train_validation, plt_loss)
    
    # evalute the model
    print("==================== compile model ========================")
    model.compile(loss=model_loss_function, optimizer=model_optimizer, metrics=model_metrics)    
    score = model.evaluate(test_data)
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])

if __name__ == "__main__":
    main()
