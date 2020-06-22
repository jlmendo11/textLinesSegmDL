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

class DataGen(keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
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
                 batch_size=10, image_shape=(512, 384), 
                 num_channels = 1, shuffle = True, 
                 toFit = True, zig_zag = False, half_outline = 0):
        """Initialization
        
        :param list_image_numbers: list of all 'label' ids to use in the generator (DEFAULT: the ones for fitting -- 1-200.tif)
        :param x_path: path to x variables location (DEFAULT: "C:\\variables_TFG\\dataset\\x")
        :param y_path: path to y variables location (DEFAULT: "C:\\variables_TFG\\dataset\\y")
        :param batch_size: batch size at each iteration (DEFAULT: 10)
        :param dim: tuple indicating image dimension (DEFAULT: 4096 x 3072)
        :param num_channels: number of image channels (DEFAULT: 1 - Black and White)
        :param shuffle: True to shuffle label indexes after every epoch (DEFAULT: True)
        :param to_fit: True to return x and y [fitting], False to return x only [prediction] (DEFAULT: True)
        :param half_outline: taking 0 it does not affect to the project. However, taking the value +1 makes this return the 
        					upper half of the outline and taking the value -1, the the other half (DEFAULT: 0)
        :param zig_zag: True if we are going to return zig_zag outline, False if just regular outline (DEFAULT: False)
        """
        self.list_image_numbers = list_image_numbers
        self.x_path = x_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.toFit = toFit
        self.zig_zag = zig_zag
        if(not zig_zag): 
            self.half_outline =  0 # if zig_zag == False there's no point in taking any half
        else:
            self.half_outline = half_outline
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
        #print("\tgetItem(" + str(batch_index) + ")")
        
        # If this is the very last batch (meaning that next one starts out of 
        # range) we are going to get only the last indexes available.
        # EXAMPLE: with a batch size of 7 and 200 elements, we have 28 full
        # batches and last batch with batch_index=28 will have only 4 elements.
        # if( (29+1)*7 > 200 ):
        #   batch_size = 200 - 29
        if(batch_index+1)*self.batch_size > len(self.list_image_numbers):
            self.batch_size = len(self.list_image_numbers) - batch_index*self.batch_size
        
        #print("Debugging1: batch_size = " + str(self.batch_size))
        
        # Extract the ids of this batch from the list.
        # EXAMPLE: with batch_index = 1 and the same situation described before,
        # a list made by list_image_numbers = range(1,201) {1,2,3...198,199,200}
        # we should store in elements_batch the sub-list {8,9,10,11,12,13,14}
        # so we will have to take list_image_numbers[7:13]
        # HOWEVER, we should take that "sub-list" from the object "indexes"
        # which will be properly shuffled
        elements_batch = self.indexes[batch_index*self.batch_size : (batch_index+1)*self.batch_size]
        
        #print("Debugging2: indexes = " + str(self.indexes))
        #print("Debugging3: elements_batch = " + str(elements_batch))
        
        if(self.toFit):
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
        if(self.num_channels == 1):
            if(self.toFit):
                x = np.empty((self.batch_size, self.image_shape[0], self.image_shape[1], self.num_channels), dtype="float32")
                y = np.empty((self.batch_size, self.image_shape[0], self.image_shape[1], self.num_channels), dtype="float32")
                
                #print("Debugging4: x, y shapes are: " + str(x.shape) + " " + str(y.shape))
                
                for i in range(0,elements_batch.size):
                    x[i,], y[i,] = self.load_single_image(self.list_image_numbers[elements_batch[i]])
                
                return x, y
            
            else:
                x = np.empty((self.batch_size, self.image_shape[0], self.image_shape[1], self.num_channels), dtype="float32")
                
                for i in range(0,elements_batch.size):
                    x[i,] = self.load_single_image(self.list_image_numbers[elements_batch[i]])
                
                return x
               
        else:
            # dtype is "bool" because I only needed black and white images
            # Feel free to implement this code by changing "dtype" if you need it
            raise NotImplementedError
            
            
    def load_single_image(self, image_number):
        """Generates one image if toFit==False and one image and its label if else
        
        :param elements_batch: list of label ids to load
        :return: batch of images
        """
        
        #print("Debugging5: load single image number: " + str(image_number))
        
        if(self.num_channels == 1):
            if(self.toFit):
                # Path
                image_number = "{:03d}".format(image_number)
                image_path = os.path.join(self.x_path, image_number) + ".npy"
                label_path = os.path.join(self.y_path, image_number) + ".npy"
                
                #print("Debugging6: looking for the images in:")
                #print("\timage_path="+image_path)
                #print("\tlabel_path="+label_path)
                
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
                x = np.reshape(x, (8*self.image_shape[0], 8*self.image_shape[1]))
                y = np.reshape(y, (8*self.image_shape[0], 8*self.image_shape[1]))
                
                x = block_reduce(x, block_size=(2, 2), func=np.max)
                x = block_reduce(x, block_size=(2, 2), func=np.min)
                x = block_reduce(x, block_size=(2, 2), func=np.max)
                y = block_reduce(y, block_size=(2, 2), func=np.max)
                y = block_reduce(y, block_size=(2, 2), func=np.min)
                y = block_reduce(y, block_size=(2, 2), func=np.max)

                x = np.reshape(x, (self.image_shape[0], self.image_shape[1], self.num_channels))
                y = np.reshape(y, (self.image_shape[0], self.image_shape[1], self.num_channels))
                #print("Debugging7: x, y shapes are: " + str(x.shape) + " " + str(y.shape))
                
                return x, y
            
            else:
                # Path
                image_number = "{:03d}".format(image_number)
                image_path = os.path.join(self.x_path, image_number) + ".npy"
                
                # Load variable
                x = np.load(image_path)
                
                # Downscaling
                x = block_reduce(x, block_size=(2, 2), func=np.max)
                x = block_reduce(x, block_size=(2, 2), func=np.min)
                x = block_reduce(x, block_size=(2, 2), func=np.max)
                
                return x
            
        else:
            # dtype is "bool" because I only needed black and white images
            # Feel free to implement this code by changing "dtype" if you need it
            raise NotImplementedError
    
    def on_epoch_end(self):     
        """Updates indexes after each epoch
        
        """
        # Create an attribute called "indexes", which is a numpy copy of list_image_number
        self.indexes = np.arange(len(self.list_image_numbers))
        # and shuffle it if shuffle==True
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
