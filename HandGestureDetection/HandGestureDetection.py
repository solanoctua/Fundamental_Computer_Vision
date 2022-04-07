import cv2, time, os
import HandTrackingMonocular as htm
import numpy as np 

def distance_between_points(point1, point2):
    return np.sqrt(np.power(point1[0]-point2[0],2) + np.power(point1[1]-point2[1],2))

folderPath = "C:/Users/asus/Desktop/ComputerVision/Hand/GesturePics"
signsPath = os.listdir(folderPath)
signList = []
for path in signsPath:
    print("{}/{}".format(folderPath, path))
    sign = cv2.imread("{}/{}".format(folderPath, path))
    signList.append(sign)
print(len(signList))

fingerPoints = [(8,6),(12,10),(16,14),(20,18)]
cam = cv2.VideoCapture(0)
frame_width, frame_height = 480,480
detector = htm.handDetector(min_detection_confidence=0.7)
prev_frame_time = 0
new_frame_time = 0
if cam.isOpened():
    ret,frame = cam.read()
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    #print("FPS: ",fps)
    output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 2.0, (frame_width, frame_height)) # 'M','J','P','G' #https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
else: 
    ret = False
while ret :
    ret,frame = cam.read()
    frame = cv2.resize(frame,(frame_width, frame_height ))
    #frame =cv2.flip(frame,-1)

    #Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    #Detect Hands
    frame = detector.detectHands(frame, draw=True)
    principalhandcoords = detector.findPixelCoords(frame, 0, draw=False)
    if (len(principalhandcoords)):
        #print(principalhandcoords)
        cv2.circle(frame, (int(principalhandcoords[4][1]),int(principalhandcoords[4][2])), 10, (0,255,255), -1) 
        cv2.circle(frame, (int(principalhandcoords[8][1]),int(principalhandcoords[8][2])), 10, (255,0,255), -1)
        cv2.line(frame,(int(principalhandcoords[4][1]),int(principalhandcoords[4][2])),(int(principalhandcoords[8][1]),int(principalhandcoords[8][2])),(0,255,0),2)

        distance = distance_between_points(principalhandcoords[4][1:], principalhandcoords[8][1:])
        midpoint = ((principalhandcoords[4][1] + principalhandcoords[8][1])//2,(principalhandcoords[4][2] + principalhandcoords[8][2])//2)
        if distance <= 20:
            cv2.circle(frame, midpoint, 10, (0,255,0), -1)
        else:
            cv2.circle(frame, midpoint, 10, (0,0,255), -1)
        cv2.putText(frame," distance:{}".format(int(distance)),midpoint,cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
        fingerPositions = []

        
        if principalhandcoords[0][2] > principalhandcoords[1][2] and principalhandcoords[0][2] > principalhandcoords[17][2]: # if hand is in upright position
            for point in fingerPoints:
                if principalhandcoords[point[0]][2] < principalhandcoords[point[1]][2]:
                    fingerPositions.append(1)
                else:
                    fingerPositions.append(0)
            if distance_between_points(principalhandcoords[4], principalhandcoords[9]) >= 30:
                fingerPositions.append(1)
            else:
                fingerPositions.append(0)

        print(fingerPositions)
        uprightFingers = fingerPositions.count(1)
        if uprightFingers > 0:
            frame[0:100,0:100] = signList[uprightFingers-1]

    cv2.putText(frame,"FPS:{}".format(int(fps)),(frame_height -80,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)# Displays fps
    #output.write(frame)
    cv2.imshow("realTimeCamera", frame)    
    key=cv2.waitKey(1)
    if key==27:
        break
cv2.destroyAllWindows()
output.release()
cam.release()
