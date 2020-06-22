This is a modification of the ICDAR 2013 Handwritting Contest Dataset.
We can found it here https://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/Protocol.html.
All credits to the original creators: https://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/Organizers.html

In this modification, the ground truth now is the outline of every text.

Also, we have now some extra information about the images: the number of lines
and the shape they have. The data is given in 2 formats: as int32 and as bool.

The x images are not included, you can find them in the ICDAR 2013 webpage (link above).

To open any of this images in python, use the next code:

##################################################
import numpy as np
from PIL import Image

x001 = Image.open(route_image_001)

y001 = np.fromfile(route_lines_001, dtype='int32')
y001 = np.reshape(y001, x001.shape)
##################################################


Mirror:

https://drive.google.com/open?id=1QO3dkzXEgPQNak9ejrG5b_m13OOWEeyw