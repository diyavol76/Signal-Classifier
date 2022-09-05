from Lora.dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram
from Lora.Lora_Dataset import Lora_H5_dataset
from Lora.dataset_preparation import ChannelIndSpectrogram
from Lora.RFF_training.Keras_Models import TripletNet, identity_loss

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.optimizers import rmsprop_experimental

import numpy as np

if __name__ == '__main__':
    file_path = r'C:\git-repos\LoRa_RFFI\esen_data.h5'
    file_path = r'D:\iot\lora\new_dataset\dataset\Diff_Days\dataset_220801.h5'

    dev_range = np.arange(0, 6, dtype=int)
    #pkt_range = np.arange(0, 750, dtype=int)
    #snr_range = np.arange(20, 80)

    LoadDatasetObj = LoadDataset()

    ChannelIndSpectrogramObj = ChannelIndSpectrogram()

    training_dataset = Lora_H5_dataset(file_path, convert_complex=True, transform=None, data_start=None,
                                                  data_end=None)
    # Load preamble IQ samples and labels.
    #data, label = LoadDatasetObj.load_iq_samples(file_path,dev_range,pkt_range)



    data_train_full = []
    label_train = []
    for t_images, t_label in training_dataset:
        data_train_full.append(t_images)

        label_train.append(t_label.numpy())
    print("shape of", np.shape(data_train_full), np.shape(label_train))

    # Add additive Gaussian noise to the IQ samples.
    #data = awgn(data, snr_range)

    # Channel independent spectogram
    data_train_full = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_train_full)
    print("enroll shape after spectogram", np.shape(data_train_full))

    TripletNetObj = TripletNet()

    print(data_train_full.shape)

    # Specify hyper-parameters during training.
    margin = 0.1
    batch_size = 32
    patience = 20

    feature_extractor = TripletNetObj.feature_extractor(data_train_full.shape)

    triplet_net = TripletNetObj.create_triplet_net(feature_extractor, margin)

    # Create callbacks during training. The training stops when validation loss
    # does not decrease for 30 epochs.
    early_stop = EarlyStopping('val_loss',
                               min_delta=0,
                               patience=
                               patience)

    reduce_lr = ReduceLROnPlateau('val_loss',
                                  min_delta=0,
                                  factor=0.2,
                                  patience=10,
                                  verbose=1)

    callbacks = [early_stop, reduce_lr]

    # Split the dasetset into validation and training sets.
    data_train, data_valid, label_train, label_valid = train_test_split(data_train_full,
                                                                        label_train,
                                                                        test_size=0.1,
                                                                        shuffle= True)

    # Create the trainining generator.
    train_generator = TripletNetObj.create_generator(batch_size,
                                                     dev_range,
                                                     data_train,
                                                     label_train)
    # Create the validation generator.
    valid_generator = TripletNetObj.create_generator(batch_size,
                                                     dev_range,
                                                     data_valid,
                                                     label_valid)

    # Use the RMSprop optimizer for training.
    # TODO opt = RMSprop(learning_rate=1e-3)
    opt = rmsprop_experimental.RMSprop(learning_rate=1e-3)
    triplet_net.compile(loss=identity_loss, optimizer=opt)

    # Start training.
    print("training started")
    history = triplet_net.fit(train_generator,
                              steps_per_epoch=1,
                              epochs=2,
                              validation_data=valid_generator,
                              validation_steps=1,
                              verbose=2,
                              callbacks=callbacks)

    feature_extractor.save('Extractor_ESEN.h5')
    print("saved")