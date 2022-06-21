import torch,cv2,time
import numpy as np

print("torch.cuda.is_available() = ",torch.cuda.is_available())
print("torch.cuda.get_device_name(0) = ",torch.cuda.get_device_name(0))
class OwlDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model(model_path)
        self.classes = self.model.names

    def load_model(self, model_path):
        # Model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_path, force_reload=True)  # or yolov5n - yolov5x6, custom
        return model
        
    def process_frame(self, frame):
        self.model.to(self.device)
        #frame = [frame]
        output = self.model(frame)
        labels, coords = output.xyxyn[0][:, -1], output.xyxyn[0][:, :-1]
        #print("object count",len(labels))
        return labels,coords
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
    
def detectFromCamera(camera_no = 0):
    cam = cv2.VideoCapture(camera_no)
    detector = OwlDetector(model_path = "best.pt")
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

if __name__ == "__main__":
    
    detectFromCamera()