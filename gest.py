import cv2
import math
import numpy as np
import pyautogui as gui

cap = cv2.VideoCapture(1)
while(cap.isOpened()):
    ret, img = cap.read()
    #cv2.resizeWindow('window1', 768,1366)
    cv2.rectangle(img, (0,0), (200,200), (0,255,0),0) ##left hand rectangle

    cv2.rectangle(img,(630,200),(430,0),(0,255,0),0) ##right hand rectangle

    crop_img = img[0:200, 0:200]
    crop_img2=img[0:200,430:630]
    

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) ## converted it to the grey scale image
    grey2 = cv2.cvtColor(crop_img2,cv2.COLOR_BGR2GRAY)


    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
    blurred2 = cv2.GaussianBlur(grey2,value,0)
    _, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(blurred2, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)
    cv2.imshow('Thresholded2', thresh2)



    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)
    else:
        contours,hierarchy=cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contours2,hierarchy1=cv2.findContours(thresh2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    cnt2= max(contours2, key = lambda x: cv2.contourArea(x))

    x, y, w, h = cv2.boundingRect(cnt)
    x2, y2, w2, h2 = cv2.boundingRect(cnt2)
    M = cv2.moments(cnt)
    M2 = cv2.moments(cnt2)
    cx = int(M['m10']/M['m00'])
    cx2 = int(M2['m10']/M2['m00'])
    cy = int(M['m01']/M['m00'])
    cy2 = int(M2['m01']/M2['m00'])
     
    cv2.circle(crop_img,(cx,cy),1,(255,0,0),6)                  
    cv2.circle(crop_img2,(cx2,cy2),1,(255,0,0),6)                  
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    cv2.rectangle(crop_img2, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 0)


    # finding convex hull
    hull = cv2.convexHull(cnt)

    hull2= cv2.convexHull(cnt2)

    drawing = np.zeros(crop_img.shape,np.uint8)
    drawing2 = np.zeros(crop_img2.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    cv2.drawContours(drawing2, [cnt2], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing2, [hull2], 0,(0, 0, 255), 0)

    hull = cv2.convexHull(cnt, returnPoints=False)
    hull2 = cv2.convexHull(cnt2, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    defects2 = cv2.convexityDefects(cnt2, hull2)
    count_defects = 0
    count_defects2=0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
    # applying Cosine Rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects

    ##LeftWindow
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)

        cv2.line(crop_img,start, end, [0,255,0], 2)

        if count_defects == 1:
            #cv2.putText(img,"I am Vipul", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            gui.moveTo(cx,cy)
        elif count_defects == 2:
            str = "This is a basic hand gesture recognizer"
            #gui.getWindow("Files").maximize()
        elif count_defects == 3:
            cv2.putText(img,"This is 4 :P", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img,"Hi!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img,"Hello World!!!", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, 2)



    cv2.drawContours(thresh2, contours2, -1, (255, 0, 0), 3)
    ##Right window
    for i in range(defects2.shape[0]):
        s2,e2,f2,d2 = defects2[i,0]

        start2 = tuple(cnt2[s2][0])
        end2 = tuple(cnt2[e2][0])
        far2 = tuple(cnt2[f2][0])

        # find length of all sides of triangle
        a2 = math.sqrt((end2[0] - start2[0])**2 + (end2[1] - start2[1])**2)
        b2 = math.sqrt((far2[0] - start2[0])**2 + (far2[1] - start2[1])**2)
        c2 = math.sqrt((end2[0] - far2[0])**2 + (end2[1] - far2[1])**2)

        # apply cosine rule here
        angle2 = math.acos((b2**2 + c2**2 - a2**2)/(2*b2*c2)) * 57

        # ignore angles > 90 and highlight rest with red dots
        if angle2 <= 90:
            count_defects2 += 1
            cv2.circle(crop_img2, far2, 1, [0,0,255], -1)

        cv2.line(crop_img2,start2, end2, [0,255,0], 2)

        if count_defects2 == 1:
            cv2.putText(img,"I am Vipul", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            #gui.moveTo(cx,cy)
        elif count_defects2 == 2:
            str = "This is a basic hand gesture recognizer"
            #gui.getWindow("Files").maximize()
        elif count_defects2 == 3:
            cv2.putText(img,"This is 4 :P", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects2 == 4:
            cv2.putText(img,"Hi!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img,"Hello World!!!", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, 2)



        

    #cv2.namedWindow('window1',cv2.WINDOW_NORMAL)
    cv2.imshow('window1',img)
    cv2.resizeWindow('window1',(600,600))
    
    k = cv2.waitKey(10)
    if k == 27:
        break