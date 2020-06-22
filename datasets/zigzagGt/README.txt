This is another modification of the ICDAR 2013 Handwritting Contest Dataset.
We can found it here https://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/Protocol.html.
All credits to the original creators: https://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/Organizers.html

In this modification, the ground truth is now the zig-zag outline version of every
image. There are two types of images, the ORIGINALS and the SIMPLIFIED
	- ORIGINALS: each line defined with number "n" will be outlined and encoded
		     with the numbers "n" for the upper outline and the number "-n"
		     for the outline below.
	- SIMPLIFIED: every positive number from the codification above has been
		      collapsed into "+1" and every negative number into "-1".
		      That way, every outline will be the combination of a "+1"
	 	      and a "-1" area.

Also, we have now some extra information about the images: the number of lines
and the shape they have. The data is given in 2 formats: as int32 and as bool

The ground truth can be open with the same code used in the "completeDataset"
(read the "README there"). The x images are not included, you can find them there.

Mirror:

https://drive.google.com/open?id=1qSeSw8UQu_Oc3NICjmU9gKWw5EPbPM_9