# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 00:22:18 2021

@author: asus123
"""
import cv2 
import numpy as np

image = cv2.imread(r"valid\photos2\DSC08341.jpg")
# cv2.imshow("if imread successfully",image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(11,11),0)
# cv2.imshow("blurred",blurred)

edged = cv2.Canny(blurred, 30, 150)
# cv2.imshow("edged",edged)
# cv2.imwrite('edge.jpg', edged)

(cnts,_) = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("There are {} protential coins in the image".format(len(cnts)))

# coins = image.copy()
# cv2.drawContours(coins, cnts, -1, (0,255,0), 2)
# cv2.imshow("coins",coins)
# cv2.waitKey(0)

#裁剪
coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0,255,0), 2)
# cv2.imshow("coins",coins)
# cv2.waitKey(0)
extracted_circle = []
for (i,circle) in enumerate(cnts):
    # circle是每个提取出来的圆
    (x,y,w,h) = cv2.boundingRect(circle)
    print("coin #{}".format(i+1))
    coin_canvas = image[y:y+h, x:x+w]
    extracted_circle.append(coin_canvas)

# Test the fack circle 
def check(c):
    x,y,z = c.shape
    if x*y<1000:
        return False
    
    if x/y>0.8 and y/x>0.8:
        return True
    else:
        return False

extracted_coins=[]
for c in extracted_circle:
    if check(c):
        extracted_coins.append(c)

for (i,c) in enumerate(extracted_coins):
    print("coin #{}".format(i+1))
    cc = np.array(c)[:,:,1]
    cc = cc.astype(np.float64)
    cc2 = np.resize(cc, (64,64))