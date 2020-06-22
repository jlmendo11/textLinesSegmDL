'''
    Created by Jose Luis Mendoza for his Final Disertation, 
    "A tool for text lines segmentation in images bases on deep learning"
    2019-2020
    University of Seville, Spain
    
    The functions coded here refer to the ICDAR2013 Handwriting Segmentation 
    Contest dataset, which we can find in the next URL (in "Resources"):
        
        http://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/
        
'''
import random, matplotlib.pyplot as plt, matplotlib as mpl
import numpy as np

def createCmap():
    """This function returns the color map that will be used to plot and save
    figures.
    
    Returns:
    cmap:the color map
    
    """
    # We take the gist_rainbow color map which already exists
    cmap = plt.cm.gist_rainbow
    # We extract the 256 colors there are in form of a list
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # We shuffle the list with this particular random seed
    random.seed(4)
    random.shuffle(cmaplist)
    # We force the first color to be white, which belongs to the background
    cmaplist[0] = ('w')
    # We create the new color map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)

    return cmap

###############################################################################
    
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

###############################################################################

def createPixelWiseBackbone(image):
    """This function creates the "Pixel-Wise Backbone" from a ground truth
    image. This new image consists in one pixel per each x coordenate and
    text-line number. This pixel represents the middle of that column for each
    text-line number.
    
    Parameters:
    image (array): an array which represents the ground truth image from the
    ICDAR2013 Handwritten Segmentation Contest dataset
    
    Returns:
    array: the Pixel-Wise Backbone
    
    """
    # The number of lines in the ground truth image (remember: a number for
    # each color + the background is the number 0)
    linesN = len(np.unique(image)) - 1
    
    # We will need this for the loop
    height, width = image.shape
    
    # The array we are going to return
    pixelWiseBackbone = np.array([])
    
    # Main loop
    for j in range(width):
        # We initialize an "all background" (all 0's) column. Then, we 
        # will fill this column with the backbone pixels. The result will
        # be appended to the array which we will return
        columnToInsert = np.zeros((height, 1)).astype(int)

        # We take a column to analyze
        columnFromImage = image[:, j]
        
        for n in range(1, linesN+1):
            # We extract the index where the column matches the number
            index = np.array(np.where(columnFromImage == n))
            
            # If there's no match, the array "index" will be empty
            if index.size > 0:
                # Take the mean and round it. Then parse to int array
                iBackboneIndex = np.around(np.mean(index)).astype(int)
                columnToInsert[iBackboneIndex] = n
                
        # When the loop is over, we will have a column to insert in the BB
        try:
            pixelWiseBackbone = np.column_stack((pixelWiseBackbone, columnToInsert))
        except:
            # The first time it will throw a ValueError Exception which we have
            # to handle
            pixelWiseBackbone = columnToInsert

    return pixelWiseBackbone
###############################################################################

def createLinearRegressionBackbone(pixelWiseBackbone):
    """This function creates the "Linear Regression Backbone" from a ground 
    truth image's "Pixel-Wise Backbone". This new image consists in the 
    regression line that goes throgh the middle of every component of the 
    text-line.
    
    Parameters:
    pixelWiseBackbone (array): an array which represents "Pixel-Wise Backbone"
    of a ground truth image returned by the function "createPixelWiseBackbone"
    
    Returns:
    array: the Linear Regression Backbone
    
    """
    # The number of lines in the ground truth image (remember: a number for
    # each color + the background is the number 0)
    linesN = len(np.unique(pixelWiseBackbone)) - 1

    # The array we are going to return
    linearRegressionBackbone = np.zeros(pixelWiseBackbone.shape)
    
    
    for n in range(1, linesN+1):
        # We extract the X and Y position of every pixel corresponding to the
        # line "n". Be careful because X and Y are given the other 
        (Y_n, X_n) = np.where(pixelWiseBackbone == n)

        # We check if the coordinates are not empty 
        if X_n.size > 0:
            # We are going to take the linear regression. The line will be 
            #    y_linearRegression = m*X_n + b
            (m, b) = np.polyfit(X_n, Y_n, 1)
            
            # Now, we have to extract the minimum and maximum values of "X" for
            # this text line. 
            X_line = np.array(range(np.amin(X_n), np.amax(X_n)))
            
            # The linear regression polynomial will be evaluated from the 
            # beggining to the very last pixel of the text line
            Y_linearRegression = np.around(np.polyval([m, b], X_line)).astype(int)
            
            # We fill the coordinates we got with the number of the line
            linearRegressionBackbone[Y_linearRegression, X_line] = n

    return linearRegressionBackbone

###############################################################################

def createOutlineBackbone(image):
    """This function creates the outline of every text line from a ground 
    truth image
    
    Parameters:
    image (array): an array which represents the ground truth image from the
    ICDAR2013 Handwritten Segmentation Contest dataset
    
    Returns:
    array: the outline image
    
    """
    # The number of lines in the ground truth image (remember: a number for
    # each color + the background is the number 0)
    linesN = len(np.unique(image)) - 1
    
    # We will need this for the loop
    height, width = image.shape
    
    # The arrays we are going to use as a help
    auxArray1 = np.array([]) # to store the maximums of each line
    auxArray2 = np.array([]) # to store the minimums of each line
    auxArray3 = np.array([]) # to store the linear regression of the max
    auxArray4 = np.array([]) # to store the linear regression of the min
    
    # The array we are going to return
    outlineBackbone = np.array([])
    
    # Main loop
    for j in range(width):
        # We initialize an "all background" (all 0's) column. Then, we 
        # will fill this column with the max and min pixels. The result will
        # be appended to the auxiliars arrays 1 and 2
        columnToInsert1 = np.zeros((height, 1)).astype(int)
        columnToInsert2 = np.zeros((height, 1)).astype(int)
        
        # We take a column to analyze
        columnFromImage = image[:, j]
        
        for n in range(1, linesN+1):
            # We extract the index where the column matches the number
            index = np.array(np.where(columnFromImage == n))
            
            # If there's no match, the array "index" will be empty
            if index.size > 0:
                # Take the max and min of the indexes
                iMaxIndex = np.amax(index)
                iMinIndex = np.amin(index)
                columnToInsert1[iMaxIndex] = n
                columnToInsert2[iMinIndex] = n
                
        # When the loop is over, we will have two columns to insert in the
        # proper auxiliar arrays
        try:
            auxArray1 = np.column_stack((auxArray1, columnToInsert1))
            auxArray2 = np.column_stack((auxArray2, columnToInsert2))
        except:
            # The first time it will throw a ValueError Exception which we have
            # to handle
            auxArray1 = columnToInsert1
            auxArray2 = columnToInsert2

    # Now, here it is the second part of the function. We will take the linear
    # regression from this arrays. We will use the upper function for this.
    auxArray3 = createLinearRegressionBackbone(auxArray1)
    auxArray4 = createLinearRegressionBackbone(auxArray2)
    
    # The formula used to create the "outlineBackbone" array is the next: we are
    # going to fill an array with the line number "n" between this two values
    #       1st value: max{auxArray1, auxArray3}
    #       2nd value: min{auxArray2, auxArray4}
    for j in range(width):
        # We initialize an "all background" (all 0's) column.
        columnToInsert = np.zeros((height, 1)).astype(int)
        
        # We take  column to analyze
        columnFromAux1 = auxArray1[:, j]
        columnFromAux2 = auxArray2[:, j]
        columnFromAux3 = auxArray3[:, j]
        columnFromAux4 = auxArray4[:, j]
        
        for n in range(1, linesN+1):
            # We extract the index where the columns match the number
            index1 = np.array(np.where(columnFromAux1 == n))
            index2 = np.array(np.where(columnFromAux2 == n))
            index3 = np.array(np.where(columnFromAux3 == n))
            index4 = np.array(np.where(columnFromAux4 == n))
            
            # We know that the indexes 3 and 4 have the same domain, and we
            # also know that, unless there's no blank space in the text line,
            # their domain is greater than the indexes 1 and 2.
            # That's why the first condition is to evaluate if index3 (or 4, we
            # don't really mind) is empty or not.
            if index3 > 0:
                if index1 > 0:
                    maxIndex = max(index1[0][0], index3[0][0])
                else: 
                    maxIndex = index3[0][0]
                if index2 > 0:
                    minIndex = min(index2[0][0], index4[0][0])
                else:
                    minIndex = index4[0][0]
                
                # Now we will loop within the values we have taken to fill the
                # column to insert afterwards
                for i in range(minIndex, maxIndex + 1):
                    columnToInsert[i] = n
                
        # When the loop is over, we will have to insert the new column
        try:
            outlineBackbone = np.column_stack((outlineBackbone, columnToInsert))
        except:
            # The first time it will throw a ValueError Exception which we have
            # to handle
            outlineBackbone = columnToInsert

    return outlineBackbone
 
###############################################################################

def createZigZagBackbone(image):
    """
    
    Parameters:
    image (array): an array which represents the ground truth image from the
    ICDAR2013 Handwritten Segmentation Contest dataset
    
    Returns:
    array: the outline image
    
    """
    # The number of lines in the ground truth image (remember: a number for
    # each color + the background is the number 0)
    linesN = len(np.unique(image)) - 1
    
    # We will need this later
    height, width = image.shape
    
    # We get both of the backbones because zigzag is created comparing them
    outlineBackboneImage = createOutlineBackbone(image)
    linearRegressionBackboneImage = createLinearRegressionBackbone(image)
    
    # The returning array
    zigZagBackbone = np.zeros(image.shape)
    
    # Main loop
    for n in range(1, linesN+1):
        # We copy the line n in a separeted array
        auxArray = np.zeros(image.shape)
        auxArray[outlineBackboneImage == n] = n
        auxArray[linearRegressionBackboneImage == n] = -n
        
        # When selecting a column from the image, we have one of these two situations:
        #   1.- We don't find any number apart from 0 in it
        #   2.- We find a sequence of numbers "n" until one unique pixel of "-n"
        #       so that we should change the numbers n below for -n
        for j in range(width):
            columnFromImage = auxArray[:, j]
            
            index = np.array(np.where(columnFromImage == -n)) 
        
            # Only 2 possibilities, etiher length of index == 1 or == 0
            if index.size == 1:
                # We take indices of the elements of the line
                index2 = np.array(np.where(columnFromImage == n))
                
                # Separate those which are beyond our "-n" index
                index2 = index2[0][index2[0] > index[0][0]]
                
                # The case of having only 1 or 2  pixel of thickness is not considered
                
                # Substitute the n with -n
                columnFromImage[index2] = -n
                
                # Update the column
                auxArray[:, j] = columnFromImage
                
        zigZagBackbone = zigZagBackbone + auxArray
        
    return zigZagBackbone


###############################################################################

def createThickBackbone(image):
    # The number of lines in the ground truth image (remember: a number for
    # each color + the background is the number 0)
    linesN = len(np.unique(image)) - 1
    
    # We will need this for the loop
    height, width = image.shape
    
    # The arrays we are going to use as a help
    auxArray1 = np.array([]) # to store the linear regression of the max
    auxArray2 = np.array([]) # to store the linear regression of the min
    
    # The array we are going to return
    thickBackbone = np.array([])
    
    # Main loop
    for j in range(width):
        # We initialize an "all background" (all 0's) column. Then, we 
        # will fill this column with the max and min pixels. The result will
        # be appended to the auxiliars arrays 1 and 2
        columnToInsert1 = np.zeros((height, 1)).astype(int)
        columnToInsert2 = np.zeros((height, 1)).astype(int)
        
        # We take a column to analyze
        columnFromImage = image[:, j]
        
        for n in range(1, linesN+1):
            # We extract the index where the column matches the number
            index = np.array(np.where(columnFromImage == n))
            
            # If there's no match, the array "index" will be empty
            if index.size > 0:
                # Take the max and min of the indexes
                iMaxIndex = np.amax(index)
                iMinIndex = np.amin(index)
                columnToInsert1[iMaxIndex] = n
                columnToInsert2[iMinIndex] = n
                
        # When the loop is over, we will have two columns to insert in the
        # proper auxiliar arrays
        try:
            auxArray1 = np.column_stack((auxArray1, columnToInsert1))
            auxArray2 = np.column_stack((auxArray2, columnToInsert2))
        except:
            # The first time it will throw a ValueError Exception which we have
            # to handle
            auxArray1 = columnToInsert1
            auxArray2 = columnToInsert2

    # Now, here it is the second part of the function. We will take the linear
    # regression from this arrays. We will use the upper function for this.
    auxArray1 = createLinearRegressionBackbone(auxArray1)
    auxArray2 = createLinearRegressionBackbone(auxArray2)
    
    # Now we fill between those two lines
    for j in range(width):
        # We initialize an "all background" (all 0's) column.
        columnToInsert = np.zeros((height, 1)).astype(int)
        
        # We take  column to analyze
        columnFromAux1 = auxArray1[:, j]
        columnFromAux2 = auxArray2[:, j]
        
        for n in range(1, linesN+1):
            # We extract the index where the columns match the number
            index1 = np.array(np.where(columnFromAux1 == n))
            index2 = np.array(np.where(columnFromAux2 == n))
            
            if index1 > 0:
                maxIndex = index1[0][0]
                minIndex = index2[0][0]
                
                # Now we will loop within the values we have taken to fill the
                # column to insert afterwards
                for i in range(minIndex, maxIndex + 1):
                    columnToInsert[i] = n
                
        # When the loop is over, we will have to insert the new column
        try:
            thickBackbone = np.column_stack((thickBackbone, columnToInsert))
        except:
            # The first time it will throw a ValueError Exception which we have
            # to handle
            thickBackbone = columnToInsert

    return thickBackbone
 

###############################################################################

def simplifyZigZag(image):
    """This function collapses all the negatives values in a zig-zag backbone
    image into -1, 0 and 1
    
    Parameters:
    image (array): the zig-zag backbone representation of an image
    
    Returns:
    array: the simplified zig zag backbone
    
    """
    newImage = np.copy(image)
    newImage[newImage<0]=-1
    newImage[newImage>0]=1
    return newImage.astype(int)
    
###############################################################################
def lineThickener(x):
    """This function takes an image and puts a pixel above and under each pixel
    with the same number of the pixel
    
    Parameters:
        image(array): the image
        
    Returns:
        array: the image thckened
    """
    image = np.copy(x)
    # We will need this later
    height, width = image.shape
    
    for i in range(width):
        indices = np.where(image[:, i] != 0)
        indices = np.array(indices[0])
        
        indicesUp = indices + 1
        image[indicesUp, i] =  image[indices, i]
        
        indicesDown = indices - 1
        image[indicesDown, i] =  image[indices, i]
        
    return image
    
    
    
    
    
    
    
    
    
    
    