# CoinsClassification
This project was created to count the number of coins and estimate the total value. With a few image of coins, we will agruementate small dataset and try to using items detection network to approach our goal. Swiss coins dataset is the only that at our disposal on internet, so we altimately choose this as our target sets.

## Methods
This is a two-stage method. 

At first, Hough Method involved, however, the methods is parameter-sensitive. so we alternatively find the edge and counter of the images, and find a boundingbox for each item. With the help of the box, we masked those small items, coins, from the big image. As we know, coins have its unique features, using this, we make a filter to extract those items that have a higher posiblity to be coins. Then we can figure out the number of the coins. 

Secondly, we leverage a lite network, mobilenet, to classify the extracted small images. With the help of deeplearning, we can finally count out the total value of the showed image. 

## Shortage Analysis
1. still background sensitive. 
2. terrible performance
3. not end to end

## Future Try
We will try a new approach, utilizing YOLOv5 serial networks. Detection networks provid an end to end framework, which has an absolute better performance. 

NOTE: worth to mention, a part of my code got reference from https://www.kaggle.com/kmader/mobilenet-classification. 
