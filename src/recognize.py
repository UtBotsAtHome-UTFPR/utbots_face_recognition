import rospy
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge
import face_recognition
import numpy as np
import cv2 

class FaceRecognizer():

    def __init__(self, new_topic_rgbImg):

        # Messages
        #self.msg_rgbImg = Image()   # Image

        # OpenCV
        self.cv_img = None           # CvImage
        self.bridge = CvBridge()

        # Flags
        self.new_rgbImg = False

        # Subscribers
        self.sub_rgbImg = rospy.Subscriber(new_topic_rgbImg, Image, self.callback_rgbImg)

        # Publishers
        self.pub_marked_imgs = rospy.Publisher("/utbots/vision/faces/image", Image, queue_size=1)

        # ROS node
        rospy.init_node('face_recognizer', anonymous=True)
        
        # Time
        self.loopRate = rospy.Rate(30)

        # Algorithm variables
        self.known_face_encodings = None
        self.known_face_names = None
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        self.train()
        self.mainLoop()



    def callback_rgbImg(self, msg):
        #self.msg_rgbImg = msg
        self.cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.new_rgbImg = True


    def train(self):
        
        # How to create a new person
        obama_image = face_recognition.load_image_file("obama.jpeg")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

        self.known_face_encodings = [obama_face_encoding]

        self.known_face_names = ["Barack Obama"]
        

    def recognize(self):
    
        #ret, frame = video_capture.read()

        small_frame = cv2.resize(self.cv_img, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame#[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)
            
            print(face_names)
                

        #self.process_this_frame = not self.process_this_frame

    def mainLoop(self):
        while rospy.is_shutdown() == False:
            # Controls speed
            self.loopRate.sleep()
            if self.new_rgbImg:
                self.new_rgbImg = False
                self.recognize()
            
            #self.pub_marked_imgs.publish(self.msg_rgbImg)
        



if __name__ == "__main__":
    FaceRecognizer(
        "/usb_cam/image_raw")