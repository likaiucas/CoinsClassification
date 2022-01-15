import cv2
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')


image = cv2.imread(r"valid\DSC08341.jpg")
scaling = 800.0/max(image.shape[0:2])
  
img_gray = cv2.resize(image, None, fx=scaling, fy=scaling)

gray = cv2.cvtColor(img_gray,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(3,3),0)


#edged = cv2.Canny(blurred, 30, 150)
#cv2.imshow("10",blurred)
#cv2.waitKey(0)

circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 30,param2 = 35, minRadius = 8, maxRadius = 60)


circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img_gray,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img_gray,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow("1",img_gray)
cv2.waitKey(0)


circles = (np.round(circles[0,:]) / scaling).astype("int")
extracted_circle = []
for ix,(x,y,r) in enumerate(circles):
    img_coin = image[y-r:y+r, x-r:x+r]
   
    img_coin = cv2.resize(img_coin, (120,120))
    extracted_circle.append(img_coin)
print("After checking, There are {} coins in the image".format(len(extracted_circle)))



