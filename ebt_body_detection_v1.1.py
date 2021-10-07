import cv2
import numpy as np

cap = cv2.VideoCapture("body.mp4")
body_cascade = cv2.CascadeClassifier("fullbody.xml")

while True:

    ret,frame = cap.read()
    if frame is None:
        break

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    body_cascade_gray = body_cascade.detectMultiScale(frame_gray,1.3,3)

    for x,y,w,h in body_cascade_gray:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow("Body Detection Video",frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

