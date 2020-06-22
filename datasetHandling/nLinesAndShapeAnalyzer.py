'''
    Created by Jose Luis Mendoza for his Final Disertation, 
    "A tool for text lines segmentation in images bases on deep learning"
    2019-2020
    University of Seville, Spain
    
    The following code makes a dataset analyzing
    
    The dataset can be found in the following URL:
        
        http://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/
'''                     
# If you feel more comfortable, then create it locally and then move it wherever 
route_shape = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\particular_test\\shape"
route_nLines = "C:\\Program Files (x86)\\ICDAR2013HandSegmCont\\particular_test\\nLines"

list_heights = []
list_widths = []
list_nLines = []

import statistics

for x in range(201,351):
    if(x != 224 and x != 327):
        route_shape_x = route_shape + "\\" + "{:03d}".format(x) + ".txt"
        route_nLines_x = route_nLines + "\\" + "{:03d}".format(x) + ".txt"
        
        f = open(route_shape_x,"r")
        shape_x = f.read().split()
        list_heights.append(int(shape_x[0]))
        list_widths.append(int(shape_x[1]))
        f.close()
        
        f = open(route_nLines_x,"r")
        nLines_x = f.read()
        list_nLines.append(int(nLines_x))
        f.close()



print("The minimum - average - maximum values of HEIGHT are:\n\t\t" + \
      str(min(list_heights)) + " - " +  str(statistics.mean(list_heights)) + " - " + str(max(list_heights)))

print("\nThe minimum - average - maximum values of WIDTH are:\n\t\t" + \
      str(min(list_widths)) + " - " + str(statistics.mean(list_widths)) + " - " + str(max(list_widths)))

print("\nThe minimum - average - maximum values of NLINES are:\n\t\t" + \
      str(min(list_nLines)) + " - " + str(statistics.mean(list_nLines)) + " - " + str(max(list_nLines)))
