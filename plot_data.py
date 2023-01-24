############################################################################
##                      Class for plotting graphs                         ##
############################################################################

import matplotlib
import matplotlib.pyplot as plt
import itertools
import numpy as np
import math

from sklearn.metrics import roc_curve, auc


class PlotData(object):
    def __init__(self):
        super(PlotData, self).__init__()
       
        # matplotlib.style.use('ggplot')

    def plot_2d(self, x, y, x_label, y_label, title, legend_arr, path_to_save):
        fig = plt.figure()
        plt.clf()
        plt.plot(x)
        plt.plot(y)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend(legend_arr, loc='best')
        plt.title(title)
        plt.savefig(path_to_save)

    def plot_model(self, model_details, path_to_save):
        plt.clf()
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # History for accuracy
        axs[0].plot(range(1, len(model_details.history['acc']) + 1),
                    model_details.history['acc'])
        axs[0].plot(range(1, len(model_details.history['val_acc']) + 1),
                    model_details.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(model_details.history[
                          'acc']) + 1), len(model_details.history['acc']) / 10)
        axs[0].legend(['train', 'validation'], loc='best')

        # History for loss
        axs[1].plot(range(1, len(model_details.history['loss']) + 1),
                    model_details.history['loss'])
        axs[1].plot(range(1, len(model_details.history['val_loss']) + 1),
                    model_details.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(model_details.history[
                          'loss']) + 1), len(model_details.history['loss']) / 10)
        axs[1].legend(['train', 'validation'], loc='best')

        # Save plot
        plt.savefig(path_to_save)

    def plot_confusion_matrix(self, cm, classes, path_to_save, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     verticalalignment='center',
                     fontsize=20,
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        ax.grid(False)
        plt.savefig(path_to_save, format='png')

    def plot_roc(self, y_true, y_scores, filename):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print('ROC AUC:', roc_auc)
        plt.figure()
        plt.clf()
        plt.plot(fpr, tpr, color='darkorange',
                 label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="best")
        plt.savefig(filename)
        plt.close()
        return roc_auc
