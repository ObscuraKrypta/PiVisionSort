import cv2
import os
import numpy as np
from sklearn.cluster import KMeans





hsv_data = []

for i ,metal in enumerate(['messing','aluminum','copper']):



    for imgpath in os.listdir(metal):
        print(len(hsv_data)+1)
        
       
        orig_image=cv2.imread(os.getcwd()+'\\'+metal+'\\'+imgpath)
        #cv2.imshow('orig',orig_image)
        imgray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.Canny(imgray,30,400)
        thresh = cv2.GaussianBlur(thresh,(3,3),0)
        #cv2.imshow('thresh',thresh)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moments=[]
        contours = list(contours)
        for cnt in contours.copy():
            if cv2.contourArea(cnt) <1000:
                contours.remove(cnt)
            else:
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                moments.append((cx,cy))
                break
        print(moments)
        new_mask=np.zeros(np.shape(orig_image)[:])    
        cv2.drawContours(new_mask, contours, -1, (255,255,255), thickness=-1)
        
        img2_fg = np.multiply(orig_image,new_mask//255)
        
        #cv2.imshow('img2fg',img2_fg)



        # START HERE
        img = img2_fg
        height, width, _ = np.shape(img)

        # reshape the image to be a simple list of RGB pixels
        image = img.reshape((height * width, 3))

        # we'll pick the 2 most common colors
        num_clusters = 2
        clusters = KMeans(n_clusters=num_clusters)
        clusters.fit(image)
        bgr=np.floor(clusters.cluster_centers_[1])


        hsv_value = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
        print('hsv value', hsv_value[0][0])
        newHsv=list(hsv_value).extend([i,metal])
        hsv_data.append(newHsv)
print(hsv_data)
np.savetxt("hsv_data.csv", 
           hsv_data,
           delimiter =", ", 
           fmt ='% s')

