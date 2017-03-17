import numpy as np
import cv2
import matplotlib.pylab as plt

from operator import mul


### Image Analysis ###

effective_Hight = 600
width_of_membrane = 21 #um
num_bin = 20
# Step 1 Read and Crop the image
img = cv2.imread('test_pore.png')
crop_img = img[:effective_Hight, :, :]
height, width, channels = crop_img.shape
scale = width_of_membrane / width

# Step 2 Convert to Grayscale and Enhance Contrast of Image
gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY )
contrast_img = cv2.equalizeHist(gray_img)

# Step 3 Apply 'averaging' filter to enhance to smooth the image
blur_kernel = (3,3)
smooth_img = cv2.blur(contrast_img, blur_kernel)

# Step 4 Apply 'OTSU method' to obtain binary image
ret,bi_img = cv2.threshold(smooth_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Step 5 Invert the binary image
inverted_img = cv2.bitwise_not(bi_img)

porosity = sum([1. for i in inverted_img.ravel() if i == 255])/ (height*width)
print 'The porosity of the membrane is {}' .format(round(porosity,2))

# Step 6 Remove isloated noise
kernel = np.ones((5,5),np.uint8)
clean_img = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, kernel)


def imclearborder(imgBW, radius):
	'''Define a function that does the same work as imclearborder in MATLAB.
	Credit: rayryeng (http://stackoverflow.com/questions/24731810/segmenting-license-plate-characters) '''

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, 0, -1)

    return imgBWcopy
    
# Step 7 Clear the blobs with connection with borders
clear_img_2 = imclearborder(clean_img, 5)
# There are some erroneous pixels at the bottom line of the img, which are supposed to be 0 (black)
clear_img_2[-1,:] = 0

### Parameters Calculation ###
contours,hierarchy = cv2.findContours(clear_img_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Compute the area of each blob inside each contour, which is the number of pixels contained in the pore blob
Area_of_Blobs = [cv2.contourArea(contours[idx]) for idx in np.arange(len(contours))]

# Compute the perimeter of each contour
Perimeter_Blobs = [cv2.arcLength(cnt, True) for cnt in contours]

# Compute the equivalent diameter based on the same area is Area defined as the diameter of a circle that has the same area as pore blob
Equivalent_diameter = [np.sqrt(4*A/np.pi) for A in Area_of_Blobs]

# Compute Shape Factor
Shape_Factor = [l**2./(4*np.pi*A) for l, A in zip(Perimeter_Blobs, Area_of_Blobs)]

# Compute Solidity
Solidity = [A/cv2.contourArea(cv2.convexHull(cnt)) for A, cnt in zip(Area_of_Blobs, contours)]

# Compute Extent, which is computed as the area of the pore blob (Ap) divided by the area of the bounding box (Ab), 
# the latter is the smallest rectangle containing the pore blob
# The area of the minimum bounding box
A_b = [reduce(mul,cv2.minAreaRect(cnt)[1]) for cnt in contours]
Extent = [a/b for a, b in zip(Area_of_Blobs, A_b)]


# Step 8 EDT transform
edt = cv2.distanceTransform(clear_img_2, cv2.cv.CV_DIST_L2, maskSize=3)

# Compute the maximum value inside of each blob in edt-transformed image
list_edt_blob_indensity = []
mask_img = np.zeros_like(edt)
# For each list of contour points...
for idx in range(len(contours)):
    # Create a mask image that contains the contour filled in
	'''Credit: rayryeng. Link: http://stackoverflow.com/questions/33234363/access-pixel-values-within-a-contour-boundary-using-opencv-in-python/42767296?noredirect=1#comment72651744_42767296'''
    cv2.drawContours(mask_img, contours, idx, color=127, -1)

    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 127)
    list_edt_blob_indensity.append(edt[pts[0], pts[1]])

# Compute effective pore diameter
D_eff = [2*(max(l)-0.5)*scale for l in lst_intensities]


### Plot the Images ###
list_imgs = [img, crop_img, contrast_img, smooth_img, bi_img, inverted_img, clean_img, edt]
fig = plt.figure()
for num, img in enumerate(list_imgs):
    y = fig.add_subplot(4,2,num+1)
    plt.imshow(y, cmap='gray')
plt.show()

### Plot the Histograms ###
list_paras = [Area_of_Blobs, Perimeter_Blobs, Equivalent_diameter, Shape_Factor, Solidity, Extent, D_eff]
fig = plt.figure()
for num, l in enumerate(list_paras):
    y = fig.add_subplot(7,1,num+1)
    y.hist(l, num_bin)
plt.show()


# Save Computed parameters as .csv files
list_paras_name = ['Area_of_Blobs', 'Perimeter_Blobs', 'Equivalent_diameter', 'Shape_Factor', 
              'Solidity', 'Extent', 'D_eff']
import csv

for para, para_name in zip(list_paras, list_paras_name):
    with open('%s.csv' %para_name, 'wb')as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(para)
