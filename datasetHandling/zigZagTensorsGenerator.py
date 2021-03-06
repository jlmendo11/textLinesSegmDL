'''
    Created by Jose Luis Mendoza for his Final Disertation, 
    "A tool for text lines segmentation in images bases on deep learning"
    2019-2020
    University of Seville, Spain
    
    The following code generates the numpy tensors used in the keras model.
    
    The dataset can be found here:
        
        http://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/
'''
from PIL import Image;
import numpy as np;
import matplotlib as mpl, matplotlib.pyplot as plt;
import random;

# Routes of the files THIS MAY CHANGE FROM ONE COMPUTER TO ANOTHER
route_images = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\images"
route_gt = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\zigzagGt\\simplified"
route_shape = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\zigzagGt\\shape"

# Routes where to save the variables
route_x_save = "C:\\variables_TFG\\zigzagTensors\\x\\"
route_y_save = "C:\\variables_TFG\\zigzagTensors\\y\\"

# Params used in the program
TENSOR_HEIGHT = 4096
TENSOR_WIDTH = 3072

def create_x():
    for x in range(1,351):
        if(x == 224 or x == 327):
            print("X, skipping the image " + str(x) + ".tif, the enumeration will be altered")
        else:    
            x2save = np.full((TENSOR_HEIGHT, TENSOR_WIDTH,1), 0).astype("int8")
            #print("X_TEST: Loading " + str(x-200) + " out of 150")
            route_x = route_images + "\\" + "{:03d}".format(x) + ".tif"
            x_array = np.logical_not(np.array(Image.open(route_x)))            
            shape_x = x_array.shape
            x_array = np.reshape(x_array, (shape_x[0], shape_x[1], 1)) # To add third dimension
            x2save[:shape_x[0], :shape_x[1], :] = x_array
            # We want the variables saved to be "201 - 348", not "201 - 350"
            # with 2 missing
            print(x)
            if(x>224):
                x = x - 1
            if(x>=327):
                x= x - 1
            print(x)
            route2save = route_x_save + "{:03d}".format(x) + ".npy"
            np.save(route2save, x2save)
            
def create_y():
    for x in range(1,351):
        # We will skip 224 and 327 for the huge dimensions they have
        if(x == 224 or x == 327):
            print("Skipping the labelling " + str(x) + ".tif.data, the enumeration will be altered")
        else:
            x2save = np.full((TENSOR_HEIGHT, TENSOR_WIDTH,1), 0).astype("int8")
            print("Y: Loading " + str(x) + " out of 350")
            route_x = route_gt + "\\" + "{:03d}".format(x) + ".tif.dat"
            route_shape_x = route_shape + "\\" + "{:03d}".format(x) + ".txt"
            f = open(route_shape_x,"r")
            shape_x = f.read().split()
            f.close()
            shape_x[0] = int(shape_x[0])
            shape_x[1] = int(shape_x[1])
            x_array = np.reshape(np.fromfile(route_x, dtype="int32").astype("int8"), (shape_x[0],shape_x[1], 1))
            # We want the variables saved to be "1 - 348", not "1 - 350 with 2 missing"
            if(x>224):
                x = x - 1
            if(x>=327):
                x= x - 1
            x2save[:shape_x[0], :shape_x[1], :] = x_array
            route2save = route_y_save + "{:03d}".format(x) + ".npy"
            np.save(route2save, x2save)
            
def createCmapZigZag(image):
    """This function return the color map that will be used to plot and save
    figures made of zig zag images
    
    Returns:
    cmap:the color map
    
    """
    colorsImage = len(np.unique(image))
    # We take the gist_rainbow color map which already exists
    cmap = plt.cm.gist_rainbow
    # We extract the 256 colors there are in form of a list
    cmaplist = [cmap(i) for i in range(cmap.N)]
    
    # We shuffle the list with this particular random seed
    random.seed(4)
    random.shuffle(cmaplist)
    
    # Take only the colorsImage first colors
    cmaplist = cmaplist[:colorsImage]
    
    # The central of all thos colors will be white (0)
    whiteIndex = int(colorsImage/2)
    
    # We force the white color to be white, which belongs to the background
    cmaplist[whiteIndex] = ('w')
    # We create the new color map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, colorsImage)

    return cmap

def create_png_to_check():
     route_x_png = "C:\\variables_TFG\\zigzagTensors\\png\\x\\"
     route_y_png = "C:\\variables_TFG\\zigzagTensors\\png\\y\\"
    
     for x in range(1, 349):
         print("Creating image of x " + "{:03d}".format(x))
         route_x = route_x_save + "{:03d}".format(x) + ".npy"
         route2save = route_x_png +"{:03d}".format(x) + ".png"
         x = np.load(route_x)
         mpl.image.imsave(route2save, x.reshape((TENSOR_HEIGHT,TENSOR_WIDTH)), cmap=mpl.cm.binary)
    
     for x in range(1, 349):
         print("Creating image of y " + "{:03d}".format(x))
         route_x = route_y_save + "{:03d}".format(x) + ".npy"
         route2save = route_y_png +"{:03d}".format(x) + ".png"
         x = np.load(route_x)
         mpl.image.imsave(route2save, x.reshape((TENSOR_HEIGHT,TENSOR_WIDTH)), cmap=createCmapZigZag(x))
    
#create_x()
#create_y()
create_png_to_check()