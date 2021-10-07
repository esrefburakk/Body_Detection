import cv2
import numpy as np

img = cv2.imread("body.jpg")
body_cascade = cv2.CascadeClassifier("fullbody.xml")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
body_cascade_gray = body_cascade.detectMultiScale(img_gray,1.18,1)

for x,y,w,h in body_cascade_gray:
    
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()