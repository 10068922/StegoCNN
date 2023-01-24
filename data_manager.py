############################################################################
##                      Image Dataset Management Class                    ##
############################################################################

from keras.preprocessing.image import ImageDataGenerator


class DataManager(object):

    def __init__(self, img_width, img_height):
        super(DataManager, self).__init__()
        self.img_width = img_width
        self.img_height = img_height

    def get_train_data(self, train_data_dir, validation_data_dir, train_batch_size, val_batch_size):
        # used to rescale the pixel values from [0, 255] to [0, 1] interval
        datagen = ImageDataGenerator(rescale=1. / 255)
        
        # Data augmentation for applying to the model
        train_datagen_augmented = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            vertical_flip=True)  # randomly flip the images

        train_datagen_augmented2 = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,            
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.4,1.5],            
            fill_mode='nearest')

        # Retrieve image datasets for training and validation
        print("Train dataset:")
        train_generator_augmented = train_datagen_augmented.flow_from_directory(
        #train_generator_augmented = train_datagen_augmented2.flow_from_directory(
            train_data_dir,
            target_size=(self.img_width, self.img_height),
            classes=['cover', 'stego'],
            batch_size=train_batch_size,
            color_mode='rgb',
            ##class_mode='binary',
            shuffle=True)

        print("Validation dataset:")
        validation_generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(self.img_width, self.img_height),
            classes=['cover', 'stego'],
            batch_size=val_batch_size,
            color_mode='rgb',
            ##class_mode='binary',
            shuffle=True)

        return train_generator_augmented, validation_generator

    # Retrieve image dataset for testing
    def get_test_data(self, test_data_dir):
        datagen = ImageDataGenerator(rescale=1. / 255)
        
        print("Test dataset:")
        test_generator = datagen.flow_from_directory(
            test_data_dir,
            target_size=(self.img_width, self.img_height),
            classes=['cover', 'stego'],
            batch_size=16,
            color_mode='rgb',
            ##class_mode='binary',
            shuffle=False)
        return test_generator
