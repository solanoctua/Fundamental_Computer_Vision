import torch,cv2,time
import numpy as np
print("torch.cuda.is_available() = ",torch.cuda.is_available())
print("torch.cuda.get_device_name(0) = ",torch.cuda.get_device_name(0))

class ObjectDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.classes = self.model.names
        

    def load_model(self):
        # Model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='github', pretrained=True)  # or yolov5n - yolov5x6, custom
        return model

    def process_frame(self, frame):
        self.model.to(self.device)
        #frame = [frame]
        output = self.model(frame)
        labels, coords = output.xyxyn[0][:, -1], output.xyxyn[0][:, :-1]
        #print("object count",len(labels))
        return labels,coords

    def yolobox_to_boundary(self, x_center, y_center, width, height):
        confidence = int(confidence*100)
        x_center = int(x_center * frame_x) 
        y_center = int(y_center * frame_y)
        width = int(width * frame_x)
        height = int(height * frame_y)
        start_point = ((x_center-width)//2, (y_center-height)//2)
        end_point = ((x_center+width)//2, (y_center+height)//2)

        return start_point, end_point

    def draw_boundaries(self, frame, labels_and_coords):
        bounding_box = 0
        box_color = (0, 255, 0) #BGR
        text_color = (255, 0, 255) #BGR
        thickness = 2
        labels, coords = labels_and_coords
        
        number_of_objects = len(labels)
        frame_y,frame_x,channels = frame.shape  # y = height = number of pixel rows, x = width = number of pixel columns
        blank = np.zeros(frame.shape, np.uint8)

        alpha = 0.1
        beta = (1.0 - alpha)
        
        for object in range(number_of_objects):
            
            x_start, y_start, x_end, y_end, confidence = coords[object]
            confidence = int(confidence*100)
            x_start = int(x_start * frame_x) 
            y_start = int(y_start * frame_y)
            x_end = int(x_end * frame_x)
            y_end = int( y_end * frame_y)
            start_point = (x_start, y_start)
            end_point = (x_end, y_end)
            #print("confidence = %",int(confidence*100))
            
            #print(f"x_center={x_center}, y_center={y_center}, width={width}, height={height}, confidence={confidence}")
            if confidence >= 0.3:
                cv2.putText(frame, self.label_to_class(labels[object]), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                cv2.putText(frame, "%"+str(confidence), (start_point[0], start_point[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                if bounding_box == 0:
    
                    cv2.rectangle(frame, start_point, end_point, box_color, thickness)
                if bounding_box == 1:
                    cv2.rectangle(blank, start_point, end_point, (0,255,0), -1)
                    cv2.addWeighted(blank, alpha, frame, beta, 0.0, frame) # to make rectangle transparent

        return frame


    def label_to_class(self, label):

        return self.classes[int(label)]

def detectFromVideo(video_path):
    cap = cv2.VideoCapture(video_path)  
    detector = ObjectDetector()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #print(f"width = {width},height = {height}, fps = {fps}")
    output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (width, height))
    if (cap.isOpened()== False):
        print("Error: cannot open the video file")
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            #Detect and draw boxes
            labels_and_coords = detector.process_frame(frame)
            frame = detector.draw_boundaries(frame,labels_and_coords)
            cv2.imshow(f"{video_path}",frame)
            output.write(frame) 
            key=cv2.waitKey(1)
            if key==27:
                break
            
        else:
            break   
    cv2.destroyAllWindows()
    output.release()
    cap.release()        
def detectFromCamera(camera_no = 0):
    cam = cv2.VideoCapture(camera_no)
    detector = ObjectDetector()
    prev_frame_time = 0
    new_frame_time = 0
    if cam.isOpened():
        ret,frame = cam.read()
        #frame_width, frame_height = (640,640)
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = cv2.VideoWriter("outputcam.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width, frame_height)) #https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
    else: 
        ret = False
    while ret :
        ret,frame = cam.read()
        #frame = cv2.resize(frame,(frame_width, frame_height ))
        cv2.imshow("RealTimeFeed",frame)

        #Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame,"FPS:{}".format(int(fps)),(15,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1,cv2.LINE_AA)#Displays fps
        #Detect and draw boxes
        labels_and_coords = detector.process_frame(frame)
        frame = detector.draw_boundaries(frame,labels_and_coords)
        #Show real time results
        cv2.imshow("RealTimeFeed",frame)
        output.write(frame) 
        key=cv2.waitKey(1)
        if key==27:
            break
    cv2.destroyAllWindows()
    output.release()
    cam.release()

def detectFromImage(Image_path):
    image = cv2.imread(Image_path)
    detector = ObjectDetector()
    #Detect and draw boxes
    labels_and_coords = detector.process_frame(image)
    result = detector.draw_boundaries(image,labels_and_coords)
    cv2.imwrite("result.png",result)
    #cv2.imshow("result",result)
   
if __name__ == "__main__":
    #detectFromImage("CatDogHuman.jpg")
    #detectFromVideo("CatDog.mp4")
    #detectFromVideo("Street.mp4")
    detectFromCamera()
    
