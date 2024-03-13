Code to retrieve speeds from linear regression in
spatiotemporal mappings of thermal runaway in Li-ion batteries.
The spatiotemporal mappings have been produced using Gabor filtering
followed by cross-correlation of the filtered frames. 

In the main, please provide: 

The path to the spatiotempial map: path = 'W:\\Data\\data_processing_mi1354\\Gabor filtering\\M50_PT_Exp3\\2\\Cross_correlations.csv'
Set true or false if you want to debug, meaning diaplay all figures as the code runs as: debug = True or False (while finding the correct Image Analysis Parameters, this is recommended) 
Image Analysis Parameters
Minimum threshold for values in the spatiotemporal map as: vmin = -500
Maximum threshold for values in the spatiotempotal map as: vmax = 900
Threshold for ehich values to segment: segmentation_threshold = 0
Minimum size of a segmented pattern in the spatiotemporal map as: minimum_blob_size = 20000
Maximum size of a segmented pattern in the spatiotemporal map as: maximum_blob_size = 200000000000
r2 value for linear fitting of lines determined on the x axis as: r2_threshold_x = 0.1
r2 value for linear fitting of lines determined on the y axis as: r2_threshold_y = 0.1
Minimum number of points in x-axis for the pattern to be condiered for linear regression as:  min_fit_line_nb_x = 50
Minimum number of points in y-axis for the pattern to be condiered for linear regression as: min_fit_line_nb_y = 50
Minimum length of the linear regression line for it to be saved as: min_length_line = 0
