# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 00:22:18 2021

@author: asus123
"""
import cv2 
import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]+x1
    y2 = dets[:, 3]+y1
    scores = dets[:, 2]*dets[:, 3]

    #每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #按照score置信度降序排序
    order = scores.argsort()[::-1]

    keep = [] #保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i) #保留该类剩余box中得分最高的一个
        #得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1] #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
       
    return keep


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
img2 = image.copy()
#裁剪
coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0,255,0), 2)
# cv2.imshow("coins",coins)
# cv2.waitKey(0)
extracted_circle = []
rects = []
for (i,circle) in enumerate(cnts):
    # circle是每个提取出来的圆
    (x,y,w,h) = cv2.boundingRect(circle)
    print("coin #{}".format(i+1))
    coin_canvas = image[y:y+h, x:x+w]
    extracted_circle.append(coin_canvas)
    draw_1=cv2.rectangle(img2, (x,y), (x+w,y+h), (0,255,0), 10)
    rects.append([x,y,w,h])
    
cv2.imwrite('NotDelect.jpg', img2)

keep = py_cpu_nms(np.array(rects), 0)
keeprect = np.array(extracted_circle)[keep]
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
for c in list(keeprect):
    if check(c):
        extracted_coins.append(c)

for (i,c) in enumerate(extracted_coins):
    print("coin #{}".format(i+1))
    cc = np.array(c)[:,:,1]
    cc = cc.astype(np.float64)
    cc2 = np.resize(cc, (64,64))


klist = list(keep)
def check2(aa):
    [x, y, w, h] = aa
    if w*h<1000:
        return False
    if w/h>0.8 and h/w>0.8:
        return True
    else:
        return False
    
img3 = img2.copy()

while klist:
    a = klist.pop()
    aa = rects[a]
    [x, y, w, h] = aa
    if check2(aa):
        draw_1=cv2.rectangle(img3, (x,y), (x+w,y+h), (255,0,0), 10)
        
cv2.imwrite('Delected.jpg', img3)
