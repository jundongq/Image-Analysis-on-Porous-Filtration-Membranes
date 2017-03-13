# Image-Analysis-on-Porous-Filtration-Membranes
This repo reproduces the image analysis on porous filtration membranes introduced in the paper: "Calculation of effective pore diameters in porous filtration membranes with image analysis (http://www.sciencedirect.com/science/article/pii/S0736584507000555)".

1. Before running the code, there are several parameters that need to be set:
(1) effective_Hight, this is the effective hight of the input image, which exclude the scale bar on the bottom of the image.
(2) scale, this is in unif of mm/pixel
(3) num_bin, this is an argument for plt.hist(input, num_bin)

2. The code computes the 7 parameters related to the porous filtration membrane: ['Area_of_Blobs', 'Perimeter_Blobs', 'Equivalent_diameter', 'Shape_Factor', 'Solidity', 'Extent', 'D_eff']. Each of them will be saved in a separate .csv file for user's preferred post-processing.
