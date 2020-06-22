'''
    Created by Jose Luis Mendoza for his Final Disertation, 
    "A tool for text lines segmentation in images bases on deep learning"
    2019-2020
    University of Seville, Spain
    
    The following code is a clarifying example about the basic handling of the
    dataset provided for the ICDAR2013 Handwriting Segmentation Contest. This
    code uses the functions defined in "groundTruthFunctions.py". The dataset
    can be found in the following URL:
        
        http://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/
'''

# Routes of the files THIS MAY CHANGE FROM ONE COMPUTER TO ANOTHER
route_images = "C:\Program Files (x86)\ICDAR2013HandSegmCont\\images"
route_lines = "C:\Program Files (x86)\ICDAR2013HandSegmCont\\gt\\lines"
route_images_001 = route_images + "\\" +  "001.tif"
route_lines_001 = route_lines + "\\" + "001.tif.dat"

# Needed imports
from groundTruthFunctions import createCmap, createPixelWiseBackbone,       \
                                 createLinearRegressionBackbone,            \
                                 createOutlineBackbone, createZigZagBackbone, \
                                 createCmapZigZag, simplifyZigZag,          \
                                 lineThickener, createThickBackbone

from PIL import Image;
import numpy as np
from collections import Counter

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt


# To plot a black and white image
def plotBlackAndWhite(image):
    plt.figure()
    plt.imshow(image, cmap = mpl.cm.binary)
    plt.axis("off")
    
# To plot a colorful image
def plotColors(image):
    plt.figure()
    plt.imshow(image, cmap=createCmap())
    plt.axis("off")
    
# To plot a colorful image in the case of ZigZag
def plotColorsZigZag(image):
    plt.figure()
    plt.imshow(image, cmap=createCmapZigZag(image))
    plt.axis("off")

'''
    ORIGINAL IMAGES ANALYSIS
'''
image001 = Image.open(route_images_001)

# To open the image in the predetermined image reproducer of the system
#image001.show()

# For the mpl.cm.binary colormap:
#       BLACK --> True
#       WHITE --> False (Background)
# Image.open return it the other way. That's the reason for the logical_not
image001_array =np.logical_not(np.array(image001))

# We investigate this array
#print("The 001 image has a shape of : " + str(image001_array.shape))

# And then we plot it 
plotBlackAndWhite(image001_array)

# Let's make a zoom to the beginning of the first text line
image001_zoom = image001_array[88:357, 138:613]
plotBlackAndWhite(image001_zoom)



'''
    GROUND TRUTH IMAGES ANALYSIS    
'''
# First, we are going to read that file and parse the result to boolean form
gt001_blackAndWhite_array = np.fromfile(route_lines_001, dtype='int32').astype('bool')

# We reshape it to the same shape of the original image and plot the result
gt001_blackAndWhite_array = np.reshape(gt001_blackAndWhite_array, image001_array.shape)
plotBlackAndWhite(gt001_blackAndWhite_array)

# In theory, this array and the one we plotted earlier are exactly the same
# We are going to check so by counting the number of pixels they do NOT have
# in common.
diff = np.logical_xor(gt001_blackAndWhite_array, image001_array)
diff_counter = 0
for x in np.nditer(diff):
    if x == True:
        diff_counter = diff_counter + 1
print("There are " + str(diff_counter) + " pixels in difference between them")
# For the 001 image there are 13 out of 2.268.189 which are not the same. This
# number is very low so we can assume they are not different.

# Now, we are going to plot the ground truth image colorfully
gt001_colors_array = np.fromfile(route_lines_001, dtype='int32')
gt001_colors_array = np.reshape(gt001_colors_array, image001_array.shape)
plotColors(gt001_colors_array)

# Let's see the composition of the ground truth image
# The "Counter" function returns a dictionary where the key is the number and 
# the value is the times is repeated.
frequencyAnalysis = Counter(np.fromfile(route_lines_001, dtype='int32'))
print("FREQUENCY ANALYSIS")
for i in sorted (frequencyAnalysis.keys()) :  
    print("\tThe number " + str(i)+ " is repeated: " + str(frequencyAnalysis[i]), end=";\n") 

# Now let's get the "Pixel-Wise Backbone" and the "Linear Regression Backbone"
# from the image and plot them
gt001_pixelWiseBackbone = createPixelWiseBackbone(gt001_colors_array)
plotColors(gt001_pixelWiseBackbone)
gt001_linearRegressionBackbone = createLinearRegressionBackbone(gt001_pixelWiseBackbone)
plotColors(gt001_linearRegressionBackbone)

# We save the file and it will be our ground truth for the CNN...
gt001_linearRegressionBackbone.astype('int32').tofile('001_groundTruth.tif.dat')

# ...and we are going to open it. Later we will save this as a plot and we will
# see the results
gt001_linearRegressionBackbone = np.fromfile("001_groundTruth.tif.dat", dtype='int32')
gt001_linearRegressionBackbone = np.reshape(gt001_linearRegressionBackbone, image001_array.shape)

# Let's parse the linearRegressionBackbone to black and white and plot it
gt001_blackAndWhite_linearRegressionBackbone = gt001_linearRegressionBackbone.astype('bool')
plotBlackAndWhite(gt001_blackAndWhite_linearRegressionBackbone)

# The linearRegressionBackbone is awesome, but the "ThickBackbone" version
# seems to fit better as a ground truth for the deep-learning problem
gt001_outlineBackbone = createOutlineBackbone(gt001_colors_array)
plotColors(gt001_outlineBackbone)

# And we can get the black and white version too
gt001_blackAndWhite_outlineBackbone = gt001_outlineBackbone.astype('bool')
plotBlackAndWhite(gt001_blackAndWhite_outlineBackbone)

# ZigZagBackbone
gt001_zigZagBackbone = createZigZagBackbone(gt001_colors_array)
plotColorsZigZag(gt001_zigZagBackbone)

# SimplifiedZigZagBackbone
gt001_simplifiedZigZagBackbone = simplifyZigZag(gt001_zigZagBackbone)
plotColorsZigZag(gt001_simplifiedZigZagBackbone)

# Thickens pixel-wise and linear regression backbone
gt001_thickenedPixelWiseBackbone = lineThickener(gt001_pixelWiseBackbone)
gt001_thickenedLinearRegressionBackbone = lineThickener(gt001_linearRegressionBackbone)

# The thick backbone of the image
gt001_thickBackbone = createThickBackbone(gt001_colors_array)
gt001_blackAndWhite_thickBackbone = gt001_thickBackbone.astype('bool')

# Finally, we are going to save some images in our workspace
mpl.image.imsave('01-image001.png', image001_array, cmap=mpl.cm.binary)
mpl.image.imsave('02-gt001_colors.png', gt001_colors_array, cmap=createCmap())
mpl.image.imsave('03-gt001_pixelWiseBackbone.png', gt001_pixelWiseBackbone, cmap=createCmap())
mpl.image.imsave('04-gt001_linearRegressionBackbone.png', gt001_linearRegressionBackbone, cmap=createCmap())
mpl.image.imsave('05-gt001_blackAndWhite_linearRegressionBackbone.png', gt001_blackAndWhite_linearRegressionBackbone, cmap=mpl.cm.binary)
mpl.image.imsave('06-gt001_outlineBackbone.png', gt001_outlineBackbone, cmap=createCmap())
mpl.image.imsave('07-gt001_blackAndWhite_outlineBackbone.png', gt001_blackAndWhite_outlineBackbone, cmap=mpl.cm.binary)
mpl.image.imsave('08-gt001_zigZagBackbone.png', gt001_zigZagBackbone, cmap=createCmapZigZag(gt001_zigZagBackbone))
mpl.image.imsave('09-gt001_simplifiedZigZagBackbone.png', gt001_simplifiedZigZagBackbone, cmap=createCmapZigZag(gt001_simplifiedZigZagBackbone))
mpl.image.imsave('10-gt001_thickenedPixelWiseBackbone.png', gt001_thickenedPixelWiseBackbone, cmap=createCmap())
mpl.image.imsave('11-gt001_thickenedLinearRegressionBackbone.png', gt001_thickenedLinearRegressionBackbone, cmap=createCmap())
mpl.image.imsave('12-thickBackbone.png', gt001_thickBackbone, cmap=createCmap())
mpl.image.imsave('13-gt001_blackAndWhite_thickBackbone.png', gt001_blackAndWhite_thickBackbone, cmap=mpl.cm.binary)

