import cv2
import os
import numpy as np
from sklearn.cluster import KMeans


import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("hsv_data.csv")
print(df.head())

inputs = df[['h','s','v']]
target = df['label']



from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(inputs, target)
##tree.plot_tree(model)
##plt.show()
print('model score', model.score(inputs,target))


##print(inputs)
##print(target)










def make_histogram(cluster):
    """
    Count the number of pixels in each cluster
    :param: KMeans cluster
    :return: numpy histogram
    """
    numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=numLabels)
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist


def make_bar(height, width, color):
    """
    Create an image of a given color
    :param: height of the image
    :param: width of the image
    :param: BGR pixel values of the color
    :return: tuple of bar, rgb values, and hsv values
    """
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv_bar[0][0]
    return bar, (red, green, blue), (hue, sat, val)


def sort_hsvs(hsv_list):
    """
    Sort the list of HSV values
    :param hsv_list: List of HSV tuples
    :return: List of indexes, sorted by hue, then saturation, then value
    """
    bars_with_indexes = []
    for index, hsv_val in enumerate(hsv_list):
        bars_with_indexes.append((index, hsv_val[0], hsv_val[1], hsv_val[2]))
    bars_with_indexes.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
    return [item[0] for item in bars_with_indexes]


#################
##cap = cv2.VideoCapture(0)
##if not cap.isOpened():
##    print("Cannot open camera")
##    exit()
##while True:
##    # Capture frame-by-frame
##    ret, orig_image = cap.read()
##    # if frame is read correctly ret is True
##    if not ret:
##        print("Can't receive frame (stream end?). Exiting ...")
##        break
##    # Our operations on the frame come here
#################


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, orig_image = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
   
    #cv2.imshow('orig',orig_image)
    imgray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.Canny(imgray,30,400)
    thresh = cv2.GaussianBlur(thresh,(3,3),0)
    #cv2.imshow('thresh',thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moments=[]
    contours = list(contours)
    for cnt in contours.copy():
        if cv2.contourArea(cnt) <100:
            contours.remove(cnt)
        else:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            moments.append((cx,cy))

    if len(contours) == 0:
        print('down here with 0 countors')


        cv2.imshow('orig image',orig_image)
        if cv2.waitKey(1) == ord('q'):
            break



        continue
    print(moments)

    for i, cnt in enumerate(contours):

    
        new_mask=np.zeros(np.shape(orig_image)[:])    
        cv2.drawContours(new_mask, [cnt], -1, (255,255,255), thickness=-1)
    
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
        

##
##        # count the dominant colors and put them in "buckets"
##        histogram = make_histogram(clusters)
##        # then sort them, most-common first
##        combined = zip(histogram, clusters.cluster_centers_)
##        combined = sorted(combined, key=lambda x: x[0], reverse=True)
##
##        # finally, we'll output a graphic showing the colors in order
##        bars = []
##        hsv_values = []
##        for index, rows in enumerate(combined):
##            bar, rgb, hsv = make_bar(100, 100, rows[1])
##            print(f'Bar {index + 1}')
##            print(f'  RGB values: {rgb}')
##            print(f'  HSV values: {hsv}')
##            hsv_values.append(hsv)
##            bars.append(bar)
##
##        # sort the bars[] list so that we can show the colored boxes sorted
##        # by their HSV values -- sort by hue, then saturation
##        sorted_bar_indexes = sort_hsvs(hsv_values)
##        color_distances=[]

##        for i in (HSV_copper,HSV_alu,HSV_messing):
##            print(hsv_values[1],np.asarray(i),'vectors')
##            
##            color_distances.append(np.linalg.norm(hsv_values[1][0]-np.asarray(i)[0]))

        hsv_value = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
        print('hsv value', hsv_value[0])
        prediction = model.predict(hsv_value[0])[0]
        if prediction==0:
            met_detected="aluminum"
        elif prediction ==1:
            met_detected ="copper"
        elif prediction == 2 :
            met_detected = "messing"
##        newHsv.extend([i,metal])
##        hsv_data.append(newHsv)
##        min_value = min(color_distances)
##        min_index = color_distances. index(min_value)
##        if min_index == 0:
##            met_detected="copper"
##        elif  min_index == 1:
##            met_detected = "aluminium"
##        elif min_index ==2 :
##            met_detected ="messing"

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig_image,met_detected,moments[i], font, 1,(255,0,255),2,cv2.LINE_AA)   
    cv2.imshow('orig image',orig_image)
##        sorted_bars = [bars[idx] for idx in sorted_bar_indexes]
    #cv2.imshow( 'fdsf',np.hstack(sorted_bars))

    #cv2.imshow('dfsf', np.hstack(bars))
##    print(f'{num_clusters} Most Common Colors')
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



##print(hsv_data)
##np.savetxt("hsv_data.csv", 
##           hsv_data,
##           delimiter =", ", 
##           fmt ='% s')

