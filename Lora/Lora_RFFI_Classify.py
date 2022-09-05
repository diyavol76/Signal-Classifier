import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Lora.Lora_Dataset as Lora_Dataset
import torch
from torch.utils.data import DataLoader
from Lora.Test_Classifier import Classifier
from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score
from Lora.Lora_Transforms import convert_to_complex_H5
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from keras.optimizers import RMSprop
from keras.optimizers import rmsprop_experimental
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, ReLU, Add, Dense, Conv2D, Flatten, AveragePooling2D


extractor= r'C:\git-repos\LoRa_RFFI\Extractor_1.h5'
extractor = r'C:\git-repos\LoRa_RFFI\Extractor_label_changed_esen.h5'
data_lrrfi_1 = r'D:\iot\data\net\lora-rffi\dataset\Train\dataset_training_aug.h5'
data_lrrfi_2 = r'D:\iot\data\net\lora-rffi\dataset\Test\dataset_seen_devices.h5'
data_esen_lrff_1=r'D:\iot\lora\new_dataset\dataset\Diff_Days\dataset_220816.h5'
data_esen_lrff_2=r'D:\iot\lora\new_dataset\dataset\Diff_Days\dataset_220817.h5'
data_esen_lrff_2=r'D:\iot\lora\new_dataset\dataset\Diff_Days\dataset_220729.h5'


if __name__ == '__main__':

    get_complex_from_H5=convert_to_complex_H5()
    register_path = data_esen_lrff_1
    enroll_dataset = Lora_Dataset.Lora_H5_dataset(register_path, convert_complex=True, transform=None,data_start=None,data_end=None)
    enroll_dataloader = DataLoader(enroll_dataset, shuffle=True)

    test_path = data_esen_lrff_2
    test_dataset = Lora_Dataset.Lora_H5_dataset(test_path, convert_complex=True, transform=None,data_start=None,data_end=None)
    test_dataloader = DataLoader(test_dataset, shuffle=True)

    classifier = Classifier()
    pred_label, true_label, acc = classifier.Lora_Classification(data_register=
                                                      enroll_dataset,
                                                      data_inference=
                                                      test_dataset,
                                                      extractor_name=
                                                      extractor )

    # Plot the confusion matrix.
    conf_mat = confusion_matrix(true_label, pred_label)
    # TODO
    test_dev_range = np.arange(0, 6, dtype=int)
    classes = test_dev_range + 1
    # TODO

    plt.figure()
    sns.heatmap(conf_mat, annot=True,
                fmt='d', cmap='Blues',
                cbar=False,
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    print('Overall accuracy = %.4f' % acc)
    plt.show()
