import cv2, time
import mediapipe as mp
import numpy as np

class handDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5 ):
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands #https://google.github.io/mediapipe/solutions/hands#python-solution-api

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = self.mp_hands.Hands(static_image_mode=self.static_image_mode,
                                        max_num_hands=self.max_num_hands, 
                                        min_detection_confidence=self.min_detection_confidence, 
                                        min_tracking_confidence=self.min_tracking_confidence)

    def detectHands(self, frame, draw=True):
        frame_width, frame_height = frame.shape[:2]
        self.handsfound = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # since hand tracking only works with RGB colorspace
        #print(handsfound.multi_hand_landmarks)
        if self.handsfound.multi_hand_landmarks is not None:
            
            for hand_landmarks in self.handsfound.multi_hand_landmarks:
                if draw:
                    cv2.putText(frame,"Hands Detected:{}".format(len(self.handsfound.multi_hand_landmarks)),(frame_width-285,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2,cv2.LINE_AA)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        else:
            print("No hands in the view")
        return frame

    def findPixelCoords(self, frame, hand_no, draw = True):
        frame_width, frame_height = frame.shape[:2]
        coords = []
        if self.handsfound.multi_hand_landmarks is not None: 
            target_hand = self.handsfound.multi_hand_landmarks[hand_no]      
            for id, landmark in enumerate(target_hand.landmark):
                    c_x, c_y = int(landmark.x * frame_width),int(landmark.y * frame_height)
                    #print("(id = {}) x = {}, y = {}".format(id, c_x, c_y) )
                    coords.append([id, c_x, c_y])  
                    if draw:
                        if id == 4:
                            cv2.circle(frame, (c_x, c_y), 10, (0,255,255), -1) 
                        elif id == 8:
                            cv2.circle(frame, (c_x, c_y), 10, (255,0,255), -1)
                        

        return coords

def main():
    cam = cv2.VideoCapture(0)
    frame_width, frame_height = 480,480

    detector = handDetector(min_detection_confidence = 0.7)
    
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
        principalhandcoords = detector.findPixelCoords(frame, 0, draw=True)
        if (len(principalhandcoords)):
            print(principalhandcoords)

        cv2.putText(frame,"FPS:{}".format(int(fps)),(15,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)# Displays fps
        #output.write(frame)
        cv2.imshow("realTimeCamera", frame)    
        key=cv2.waitKey(1)
        if key==27:
            break
    cv2.destroyAllWindows()
    output.release()
    cam.release()

if __name__ == "__main__":
    main()