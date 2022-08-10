from cgitb import grey
import enum
import re
import cv2 
import numpy as np


cap =cv2.VideoCapture("highway.mp4")
min_width_rectangle= 80  #min rectangle dimension
min_height_rectangle = 80 
algo= cv2.createBackgroundSubtractorMOG2()


def center(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect = []
offset=3           #offset for touching the line error
countkiline_pos = 520
counterfinal=0           #count vehicle

while True:
    success, frame1=cap.read()
    grey1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey1,(3,3),5)
    
    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilated= cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    dilated= cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel)
    
    countershape,h  = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow('ProjectOOOOOO', dilated)
    #line draw krenge
    cv2.line(frame1,(200,countkiline_pos),(990,countkiline_pos),(0,0,255),3)

    #rect draw around obj

    for (i,channel) in enumerate(countershape):
        (x,y,w,h)= cv2.boundingRect(channel)
        validate_count = (w>=min_width_rectangle) and (h>= min_height_rectangle) 
        if not validate_count:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        centerg = center(x,y,w,h)
        detect.append(centerg)
        cv2.circle(frame1,centerg,4, (0,0,255),-1)

        for (x,y) in detect:
            if y<(countkiline_pos + offset) and y>(countkiline_pos - offset) :
                counterfinal+=1
            cv2.line(frame1,(200,countkiline_pos),(990,countkiline_pos),(0,0,0),3)
            detect.remove((x,y))
            print("Vehicle Counter"+str(counterfinal))

    cv2.putText(frame1,"Vehicle counter : "+str(counterfinal),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)

    if success:
        cv2.imshow('ProjectOOOOOO',frame1)
        if cv2.waitKey(10) == 13:
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()