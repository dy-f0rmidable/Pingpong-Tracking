# Nguyen Dang Duy - 21146377
# Ngo Quang Hoa - 21146


#----------------------IMPORT THE LIBRARY------------------------#
import numpy as np
import cv2 as cv
import time
from collections import deque

#---------------------LOAD THE VIDEO FOR THE PROJECT------------------------#
cap = cv.VideoCapture("pingpong2.mp4")
img = cv.imread('ground.png')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
#---------------------DEFINE VARIABLE-----------------------#
# Creates a double-ended queue (a.k.a deque) named pts with a maximum length of 50 to store a fixed-size history of points or values for motion tracking.
pts = deque(maxlen=50)  
count = 0
IN = 0
OUT = 0
contour_processed = False

while True:
    timer = cv.getTickCount()
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer) 
    ret, frame = cap.read()
    w = frame.shape[1]
    h = frame.shape[0]
    print("resolution",w,h)
    # Crop video for the ground
    # Gframe = frame[180:,:,:]
    Gframe = frame[520:,:,:]
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

#----------------------PROCESS THE GROUND FOR THE IN/OUT DETECTION BY DIVIDE THE GROUND REGION FRAME------------------------#
    gray = cv.cvtColor(Gframe, cv.COLOR_BGR2GRAY)
    blur_Ground = cv.medianBlur(gray, 3)
    bGround = cv.threshold(blur_Ground, 169, 255, cv.THRESH_BINARY)[1]
    f_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11),(-1,-1))
    n_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1,1),(-1,-1))    
    morph = cv.morphologyEx(bGround,cv.MORPH_OPEN, n_kernel,iterations=2) 
    morph = cv.morphologyEx(morph,cv.MORPH_CLOSE, f_kernel,iterations=2)
    contours, hierarchy = cv.findContours(morph,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,cnts in enumerate(contours):
        x,y,w,h = cv.boundingRect(cnts)

        # Approximate the contour with fewer vertices for smoother lines
        approx = cv.approxPolyDP(cnts, 0.01 * cv.arcLength(cnts, True), True)  #0.01 is old metric
        for g in range(len(approx) - 1):

            # Draw line between consecutive points
            start_point = (approx[g][0][0], approx[g][0][1])
            end_point = (approx[g + 1][0][0], approx[g + 1][0][1])
            cv.line(frame, start_point, end_point, (0, 255, 0), 2)

    # Get IN region and OUT region for checking
    inGround = Gframe[:,:start_point[0],:]
    w_in = inGround.shape[1]
    h_in = inGround.shape[0]
    outGround = Gframe[:,start_point[0]:,:]
    w_out = outGround.shape[1]
    h_out = outGround.shape[0]

#----------------------DETECT THE PINGPONG BY HSV METHOD------------------------#
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    low_hsv = (10,80,10)
    high_hsv = (25,255,255)
    mask = cv.inRange(hsv, low_hsv, high_hsv)
    b_img = cv.erode(mask, None, iterations=2)
    contour, hierachy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    center = None
    trueColor = (10,255,0)
    for cnt in contour:
        x,y,w,h = cv.boundingRect(cnt)
        contour_area = cv.contourArea(cnt)
        (xo,yo),radius = cv.minEnclosingCircle(cnt)
        circle_area = 3.14*(radius**2)      

#--------------------CHECK THE IN/OUT DROP POINTS OF THE PINGPONG----------------------#
        if yo >= h_in and xo <= w_in:
            if not contour_processed:                
                IN+=1
                count+=1
                contour_processed = True
        elif yo >= h_out and xo > w_in:
             if not contour_processed:  
                trueColor = (0,0,255)            
                OUT+=1               
                contour_processed = True
        else:
            contour_processed = False

        # Draw contour for the pingpong   
        if (contour_area/circle_area) > 0.2:
            center = (int(xo),int(yo))
            radius = int(radius)
            cv.circle(frame,center,radius,trueColor,3)
            # Put text of pingpong coordinate for visualizer
            cv.putText(frame, f"x= {int(xo)}, y= {int(yo)}", (int(xo),int(yo)), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)           

#----------------------VISUALIZE THE TRACKING OF THE PINGPONG------------------------#
    # Draw the tracking line by drawing a line between 2 queue of deque so from that we can have a smooth line of tracking
    if len(pts) > 3:
        for j in range(1, len(pts)):
            if pts[j - 1] is None or pts[j] is None:
                continue
            diemDau = pts[j - 1]
            diemCuoi = pts[j]
            tracklines = cv.line(frame, pts[j - 1], pts[j], trueColor, 3)
            cv.putText(frame, f"x= {int(diemCuoi[0])}, y= {int(diemCuoi[1])}",(int(diemCuoi[0])+10,int(diemCuoi[1])+10), cv.FONT_HERSHEY_DUPLEX,0.5,(0,0,0),1) 
    pts.appendleft(center)

#---------------------VISUALIZE THE PROJECT----------------------#
    cv.putText(frame, "FPS : " + str(int(fps)), (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv.putText(frame, "Coordinate of pingpong : " + str(center), (20, 75), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv.putText(frame,"PINGPONG VAR", (1120, 40), cv.FONT_HERSHEY_DUPLEX,1,(0, 0, 0),2)
    cv.putText(frame,"IN : "+ str(IN), (1200, 80), cv.FONT_HERSHEY_DUPLEX,1, (0, 0, 0), 2)
    cv.putText(frame,"OUT : "+ str(OUT), (1200, 115), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

#----------------------DISPLAY THE RESULT FRAME-----------------------#
    cv.imshow('pingpong', frame)
    cv.imshow('ground',img)
    cv.imshow('inGround', inGround)
    cv.imshow('outGround', outGround)
    key = cv.waitKey(16)
    if key == ord(" "):
	    cv.waitKey(0)
    elif key == ord('q'):
        break

# Release the capture when everything is done
cap.release()
cv.destroyAllWindows()