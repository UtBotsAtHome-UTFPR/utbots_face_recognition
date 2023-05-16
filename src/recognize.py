import rospy
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge
import face_recognition
import numpy as np
import cv2
import os
import os.path
import pickle

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
        self.process_this_frame = True

        self.load_train_data()

        self.mainLoop()



    def callback_rgbImg(self, msg):
        #self.msg_rgbImg = msg
        self.cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.new_rgbImg = True

    def load_train_data(self):

        
        ''' Model path, turn into a variable '''
        with open('src/face_recognition/trained_knn_model.clf', 'rb') as f:
            self.knn_clf = pickle.load(f)

    
    def recognize(self):

        self.face_locations = face_recognition.face_locations(self.cv_img)
        self.face_encodings = face_recognition.face_encodings(self.cv_img, self.face_locations)

        # Recognizing bit, move away from here
        if len(self.face_locations) == 0:
            return []

        closest_distances = self.knn_clf.kneighbors(self.face_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(self.face_locations))]

        print([(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.knn_clf.predict(self.face_encodings), self.face_locations, are_matches)])

    # Recognize people from a single photo with the subject's name
    def train_simpler(self):
        
        obama_image = face_recognition.load_image_file("obama.jpeg")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

        self.known_face_encodings = [obama_face_encoding]

        self.known_face_names = ["Barack Obama"]


    def recognize_simpler(self):

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(self.cv_img)
        self.face_encodings = face_recognition.face_encodings(self.cv_img, self.face_locations)

        face_names = []
        for face_encoding in self.face_encodings:
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
        
        #self.show_frame(self.face_locations)
                

    
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