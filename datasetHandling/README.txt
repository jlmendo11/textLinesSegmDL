This folder contains python scripts which purpose is:

- datasetHandlingExample.py             : clarifying example of how to handle ICDAR 2013 dataset. It uses "groundTruthFunctions.py".
- groudTruthFunctions.py                : contains many functions used for creating multiple versions of the dataset.
- thickBackboneGroundTruthGenerator.py  : used to generate the 350 images of the dataset to "thickBackbone" versions. The results can also be found in "datasets" folder.
- zigZagBackboneGroundTruthGenerator.py : used to generate the 350 images of the dataset to "zigZagBackbone" versions. The results can also be found in "datasets" folder.
- nLinesAndShapeAnalyzer.py             : code used to determine statistics of height, width and number of lines from the dataset. An analysis has been added in "nLinesAndShapeAnalysis.txt"
- thickBacboneTensorsGenerator.py       : used to generate the numpy tensor variables in order to speed training. The results can also be found in "tensors" folder. 
- zigZagTensorsGenerator.py             : used to generate the numpy tensor variables in order to speed training. The results can also be found in "tensors" folder. 
