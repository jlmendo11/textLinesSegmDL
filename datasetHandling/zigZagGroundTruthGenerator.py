'''
    Created by Jose Luis Mendoza for his final course Disertation, 
    "A tool for text lines segmentation in images bases on deep learning"
    2019-2020
    University of Seville, Spain
    
    The following code generates the ground truth of our FCN when executed.
    It parses every image from the ground truth provided by the ICDAR2013 
    Handwriting Segmentation Contest and it converts it into our particular
    ground truth, which consists in the "Linear Regression Backbone" version
    
    The dataset can be found in the following URL:
        
        http://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/
'''
from PIL import Image
import numpy as np
from groundTruthFunctions import createZigZagBackbone, simplifyZigZag, createCmapZigZag
import matplotlib as mpl
# Predetermined paths
'''
destination_route_int32 = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\particular_gt\\int32"
destination_route_bool = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\particular_gt\\bool"
destination_route_shape = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\particular_gt\\shape"
destination_route_nLines = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\particular_gt\\nLines"
'''

# If you feel more comfortable, then create it locally and then move it wherever 
destination_route_original = "particular_gt\\original"
destination_route_simplified = "particular_gt\\simplified"
destination_route_shape = "particular_gt\\shape"
destination_route_nLines = "particular_gt\\nLines"
destination_route_images = "particular_gt\\images"

# route_lines = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\gt\\lines"
# route_images = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\images"
route_lines = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\native_test\\gt_lines_test"
route_images = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\native_test\\images_test"

###############################################################################
# This first part parses the Bengali ground truth images to the same format
# followed by the rest of images
'''
from collections import Counter
for x in range(151, 201):
    print("Parsing Bengali TRAINING number " + str(x))
    textLineNumber = 0
    route_x = route_lines + "\\" + "{:03d}".format(x) + ".tif.dat"
    array_x = np.fromfile(route_x, dtype='int32')
    frequencyAnalysis = Counter(array_x)
    for i in sorted (frequencyAnalysis.keys()):
        array_x[array_x == i] = textLineNumber
        textLineNumber = textLineNumber + 1
    array_x.astype('int32').tofile(route_x)

for x in range(301, 351):
    print ("Parsing Bengali TEST number " + str(x))
    textLineNumber = 0
    route_x = route_lines + "\\" + "{:03d}".format(x) + ".tif.dat"
    array_x = np.fromfile(route_x, dtype='int32')
    frequencyAnalysis = Counter(array_x)
    for i in sorted (frequencyAnalysis.keys()):
        array_x[array_x == i] = textLineNumber
        textLineNumber = textLineNumber + 1
    array_x.astype('int32').tofile(route_x)
'''
    
###############################################################################    
for x in range(201, 351):
    print(x)
    
    route_x = route_lines + "\\" + "{:03d}".format(x) + ".tif.dat"
    route_image_x = route_images + "\\" + "{:03d}".format(x) + ".tif"
    destination_x_original = destination_route_original + "\\" + "{:03d}".format(x) + ".tif.dat"
    destination_x_simplified = destination_route_simplified + "\\" + "{:03d}".format(x) + ".tif.dat"
    destination_x_shape = destination_route_shape + "\\" + "{:03d}".format(x) + ".txt"
    destination_x_nLines = destination_route_nLines + "\\" + "{:03d}".format(x) + ".txt"
    
    name_original_png = destination_route_images + "\\" + "{:03d}".format(x) + "original.png"
    name_simplified_png = destination_route_images + "\\" + "{:03d}".format(x) + "simplified.png"
    
    # Extract the shape of the real image
    image_x = Image.open(route_image_x)
    shape_x = np.array(image_x).shape
    
    # We save the shape in a text file
    f = open(destination_x_shape,"w")
    shape_tofile_x = str(shape_x[0]) + " " + str(shape_x[1])
    f.write(shape_tofile_x) # write the tuple into a file
    f.close() # close the file
    # Read the binary files
    array_x = np.fromfile(route_x, dtype='int32')
    array_x = np.reshape(array_x, shape_x)
    
    # Save the number of lines in a different text file
    nLines_x= len(np.unique(array_x)) - 1
    f = open(destination_x_nLines,"w")
    f.write(str(nLines_x))
    f.close()

    # Create the targeted array
    zigZagBacbone = createZigZagBackbone(array_x)
    simplifiedZigZagBackbone = simplifyZigZag(zigZagBacbone)
    
    # Save the new binary file in 2 different forms
    zigZagBacbone.astype('int32').tofile(destination_x_original)
    simplifiedZigZagBackbone.astype('int32').tofile(destination_x_simplified)   
    
    mpl.image.imsave(name_original_png, zigZagBacbone, cmap=createCmapZigZag(zigZagBacbone))
    mpl.image.imsave(name_simplified_png, simplifiedZigZagBackbone, cmap=createCmapZigZag(simplifiedZigZagBackbone))


    