'''
    Created by Jose Luis Mendoza for his Disertation, 
    "A tool for text lines segmentation in images bases on deep learning"
    2019-2020
    University of Seville, Spain
    
    The following code is used by the CNN for taking a batch of images and labels
    to be processed.
    
    The dataset can be found here:
        
        http://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/
'''

import keras
import os
import numpy as np
from skimage.measure import block_reduce
from PIL import Image

class DataGen(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    The images will be downsampled to half the resolution saved.
    """
    # Routes where the variables are. Obviously, this would vary between implementations
    route_x_train = "C:\\variables_TFG\\x_train\\"
    route_y_train = "C:\\variables_TFG\\y_train\\"
    route_x_test = "C:\\variables_TFG\\x_test\\"
    route_y_test = "C:\\variables_TFG\\y_test\\"
    
    # This is the shape of the images
    IMAGES_HEIGHT = 4096
    IMAGES_WIDTH = 3072
    
    def __init__(self, list_image_numbers = range(1,201), 
                 x_path="C:\\variables_TFG\\dataset\\x", 
                 y_path="C:\\variables_TFG\\dataset\\y",
                 batch_size=10, image_shape=(2048, 1536), 
                 num_channels = 1, shuffle = True, 
                 to_fit = True, zig_zag = False, half_outline = 0,
                 data_augmentation=True, debugging=False):
        """Initialization
        
        :param list_image_numbers: list of all 'label' ids to use in the generator (DEFAULT: the ones for fitting -- 1-200.tif)
        :param x_path: path to x variables location (DEFAULT: "C:\\variables_TFG\\dataset\\x")
        :param y_path: path to y variables location (DEFAULT: "C:\\variables_TFG\\dataset\\y")
        :param batch_size: batch size at each iteration (DEFAULT: 10)
        :param dim: tuple indicating image dimension (DEFAULT: 4096 x 3072)
        :param num_channels: number of image channels (DEFAULT: 1 - Black and White)
        :param shuffle: True to shuffle label indexes after every epoch (DEFAULT: True)
        :param to_fit: True to return x and y [fitting], False to return x only [prediction] (DEFAULT: True)
        :param zig_zag: True if we are going to return zig_zag outline, False if just regular outline (DEFAULT: False)
        :param half_outline: taking 0 it does not affect to the project. However, taking the value +1 makes this return the 
                            upper half of the outline and taking the value -1, the the other half (DEFAULT: 0)
        :param data_augmentation: when True, 3 images are returned instead of only 1, being the 2 others the 5 degrees rotation of the original (DEFAULT: True)
        :param debugging: True to show debugging messages (DEFAULT: False)
        """
        self.list_image_numbers = list_image_numbers
        self.x_path = x_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.zig_zag = zig_zag
        if(not zig_zag): 
            self.half_outline =  0 # if zig_zag == False there's no point in taking any half
        else:
            self.half_outline = half_outline
        self.data_augmentation = data_augmentation
        self.debugging = debugging
        self.on_epoch_end() # to generate the indexes variables and shuffling 
        
        
    def __len__(self):
        """Denotes the number of batches per epoch
        
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_image_numbers) / self.batch_size))
    
    def __getitem__(self, batch_index):
        """Generate one batch of data
        
        :param batch_index: index of the batch BEGINNING WITH 0
        :return: x and y when fitting. x only when predicting
        """
        if(self.debugging):
            print("Function: getItem(" + str(batch_index) + ")")
        
        # If this is the very last batch (meaning that next one starts out of 
        # range) we are going to get only the last indexes available.
        # EXAMPLE: with a batch size of 7 and 200 elements, we have 28 full
        # batches and last batch with batch_index=28 will have only 4 elements.
        # if( (28+1)*7 > 200 ):
        #   batch_size = 200 - 28*7 = 4
        if(batch_index+1)*self.batch_size > len(self.list_image_numbers):
            self.batch_size = len(self.list_image_numbers) - batch_index*self.batch_size
        
        
        # Extract the ids of this batch from the list.
        # EXAMPLE: with batch_index = 1 and the same situation described before,
        # a list made by list_image_numbers = range(1,201) {1,2,3...198,199,200}
        # we should store in elements_batch the sub-list {8,9,10,11,12,13,14}
        # so we will have to take list_image_numbers[7:13]
        # HOWEVER, we should take that "sub-list" from the object "indexes"
        # which will be properly shuffled
        elements_batch = self.indexes[batch_index*self.batch_size : (batch_index+1)*self.batch_size]
        
        if(self.debugging):
            print("Debugging1: batch_size = " + str(self.batch_size))
            print("Debugging2: indexes = " + str(self.indexes))
            print("Debugging3: elements_batch = " + str(elements_batch))
        
        if(self.to_fit):
            x, y = self.__load_batch(elements_batch)
            #print("\t\tfin de getItem(" + str(batch_index) + ")")
            return x, y
        else:
            x = self.__load_batch(elements_batch)
            return x
    
    def __load_batch(self, elements_batch):
        """Generates an array containing images and labels of a batch
        
        :param elements_batch: list of label ids to load
        :return: batch of images and labels
        """
        # If data augmentation is activated, we will need 3 more images
        if(self.data_augmentation):
            images_to_load = 3*self.batch_size
        else:
            images_to_load = self.batch_size

        if(self.to_fit):
            x = np.empty((images_to_load, self.image_shape[0], self.image_shape[1], self.num_channels), dtype="float32")
            y = np.empty((images_to_load, self.image_shape[0], self.image_shape[1], self.num_channels), dtype="float32")
            
            if(self.debugging):
                print("Debugging4: x, y shapes are: " + str(x.shape) + " " + str(y.shape))
            
            if(self.data_augmentation):
                for i in range(0,elements_batch.size):
                    # load single image will return 3 images from each one
                    aux_x, aux_y = self.load_single_image(self.list_image_numbers[elements_batch[i]])
                    for j in range(0, 3):
                        x[3*i+j,], y[3*i+j,] =aux_x[j,], aux_y[j,]

            else:
                for i in range(0,elements_batch.size):
                    x[i,], y[i,] = self.load_single_image(self.list_image_numbers[elements_batch[i]])
            
            return x, y
        
        else:
            x = np.empty((images_to_load, self.image_shape[0], self.image_shape[1], self.num_channels), dtype="float32")
            
            if(self.debugging):
                print("Debugging4: x shape is: " + str(x.shape) )

            if(self.data_augmentation):
                for i in range(0,elements_batch.size):
                    # load single image will return 3 images from each one
                    aux_x = self.load_single_image(self.list_image_numbers[elements_batch[i]])
                    for j in range(0, 3):
                        x[3*i+j,] =aux_x[j,]

            else:
                for i in range(0,elements_batch.size):
                    x[i,] = self.load_single_image(self.list_image_numbers[elements_batch[i]])
            
            return x
            
            
    def load_single_image(self, image_number):
        """Generates one image if to_fit==False and one image and its label if else
        
        :param elements_batch: list of label ids to load
        :return: batch of images
        """
        
        if(self.debugging):
            print("Debugging5: load single image number: " + str(image_number))
        
        if(self.to_fit):
            # Path
            image_number = "{:03d}".format(image_number)
            image_path = os.path.join(self.x_path, image_number) + ".npy"
            label_path = os.path.join(self.y_path, image_number) + ".npy"
            
            if(self.debugging):
                print("Debugging6: looking for the images in:")
                print("\timage_path="+image_path)
                print("\tlabel_path="+label_path)
            
            # Load variable
            x = np.load(image_path).astype('float32')
            y = np.load(label_path).astype('float32')

            if(self.zig_zag):
                if(self.half_outline != 0):
                    # If half_outline = 1 --> upper half || half_outline = -1 --> other half
                    y =  np.floor((self.half_outline * y + 1) / 2)

            else:
                y = np.absolute(y)

            # Downscaling
            x = np.reshape(x, (2*self.image_shape[0], 2*self.image_shape[1]))
            y = np.reshape(y, (2*self.image_shape[0], 2*self.image_shape[1]))
            
            x = block_reduce(x, block_size=(2, 2), func=np.max)

            y = block_reduce(y, block_size=(2, 2), func=np.max)


            x = np.reshape(x, (self.image_shape[0], self.image_shape[1], self.num_channels))
            y = np.reshape(y, (self.image_shape[0], self.image_shape[1], self.num_channels))


            # Data augmentation
            if(self.data_augmentation):
                x_image = Image.fromarray(np.reshape(x, (self.image_shape[0], self.image_shape[1])))
                y_image = Image.fromarray(np.reshape(y, (self.image_shape[0], self.image_shape[1])))
                
                # Rotate +5 degrees
                x_pos5 = np.array(Image.Image.rotate(x_image, 5))
                y_pos5 = np.array(Image.Image.rotate(y_image, 5))
                
                # Rotate -5 degrees
                x_neg5 = np.array(Image.Image.rotate(x_image, -5))
                y_neg5 = np.array(Image.Image.rotate(y_image, -5))

                # Array to return
                x_augmentated = np.empty((3, self.image_shape[0], self.image_shape[1], self.num_channels), dtype = 'float32')
                y_augmentated = np.empty((3, self.image_shape[0], self.image_shape[1], self.num_channels), dtype = 'float32')

                # Fill the arrays
                x_augmentated[0,] = np.reshape(x_pos5, (self.image_shape[0], self.image_shape[1], self.num_channels))
                x_augmentated[1,] = x
                x_augmentated[2,] = np.reshape(x_neg5, (self.image_shape[0], self.image_shape[1], self.num_channels))
                y_augmentated[0,] = np.reshape(y_pos5, (self.image_shape[0], self.image_shape[1], self.num_channels))
                y_augmentated[1,] = y
                y_augmentated[2,] = np.reshape(y_neg5, (self.image_shape[0], self.image_shape[1], self.num_channels))

                if(self.debugging):
                    print("Debugging7: x_augmentated, y_augmentated shapes are: " + str(x_augmentated.shape) + " " + str(y_augmentated.shape))

                return x_augmentated, y_augmentated

            else:
                if(self.debugging):
                    print("Debugging7: x, y shapes are: " + str(x.shape) + " " + str(y.shape))
                return x, y

            
        # If "to_fit == False" then we do the same but only with x
        else:
            # Path
            image_number = "{:03d}".format(image_number)
            image_path = os.path.join(self.x_path, image_number) + ".npy"
            
            if(self.debugging):
                print("Debugging6: looking for the images in:")
                print("\timage_path="+image_path)
            
            # Load variable
            x = np.load(image_path).astype('float32')

            # Downscaling
            x = np.reshape(x, (2*self.image_shape[0], 2*self.image_shape[1]))
            
            x = block_reduce(x, block_size=(2, 2), func=np.max)

            x = np.reshape(x, (self.image_shape[0], self.image_shape[1], self.num_channels))

            # Data augmentation
            if(self.data_augmentation):
                x_image = Image.fromarray(np.reshape(x, (self.image_shape[0], self.image_shape[1])))
                
                # Rotate +5 degrees
                x_pos5 = np.array(Image.Image.rotate(x_image, 5))
                
                # Rotate -5 degrees
                x_neg5 = np.array(Image.Image.rotate(x_image, -5))

                # Array to return
                x_augmentated = np.empty((3, self.image_shape[0], self.image_shape[1], self.num_channels), dtype = 'float32')

                # Fill the arrays
                x_augmentated[0,] = np.reshape(x_pos5, (self.image_shape[0], self.image_shape[1], self.num_channels))
                x_augmentated[1,] = x
                x_augmentated[2,] = np.reshape(x_neg5, (self.image_shape[0], self.image_shape[1], self.num_channels))

                if(self.debugging):
                    print("Debugging7: x_augmentated shape is: " + str(x_augmentated.shape))

                return x_augmentated

            else:
                if(self.debugging):
                    print("Debugging7: x shape is: " + str(x.shape))
                return x
    
    def on_epoch_end(self):     
        """Updates indexes after each epoch
        
        """
        # Create an attribute called "indexes", which is a numpy copy of list_image_number
        self.indexes = np.arange(len(self.list_image_numbers))
        # and shuffle it if shuffle==True
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
