#!/usr/bin/python3

# Node for training the network for a new person

import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import timeit

class Trainer:

    def __init__(self, save_name="trained_knn_model.clf", n: int=None, face_crop: bool=False):

        self.train_dir = os.path.realpath(os.path.dirname(__file__)) + "/../faces"
        self.model_save_path = os.path.realpath(os.path.dirname(__file__)) + "/../trained_models/" + save_name
        self.knn_algo = 'ball_tree'

        # OpenCV
        self.cv_img = None           # CvImage
        self.bridge = CvBridge()

        # Messages
        self.msg_command = String()

        # Subscribers
        self.sub_command = rospy.Subscriber("/utbots/task_manager/manager_commands", String, self.callback_commands)

        # Publishers
        self.pub_current_img = rospy.Publisher("/utbots/vision/faces/image", Image, queue_size=1)
        self.pub_enable_face = rospy.Publisher("/utbots/vision/faces/enable", Bool, queue_size=1)
        self.pub_speech = rospy.Publisher("/utbots/voice/tts/robot_speech", String, queue_size=1)

        # ROS node
        rospy.init_node('face_recognizer_trainer', anonymous=True)
        
        # Time
        self.loopRate = rospy.Rate(30)

        # Eventually update verbose for making debugging easier (when it uses voice commands and so on)
        self.verbose = True # If set to True talks about its actions
        self.should_face_crop = face_crop # If set to true, crops the images being taken to have only a face on them

        # Algorithm variables
        self.names = []
        self.face_encodings = []
        self.n_neighbors = n
        
        self.mainLoop()

    def callback_commands(self, msg):
        self.msg_command = msg

    def load_faces(self):

        rospy.loginfo("[TRAIN] Loading faces for training")
        self.pub_speech.publish("Loading faces for training")

        # Loop through each person in the training set
        for class_dir in os.listdir(self.train_dir):
            if not os.path.isdir(os.path.join(self.train_dir, class_dir)):
                continue
        
            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(self.train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)

                face_bounding_boxes = [(1, image.shape[1] - 1, image.shape[0] - 1, 1)]
                #face_bounding_boxes = face_recognition.face_locations(image)
                #print(face_recognition.face_locations(image))
                
                #print(image.shape)
                if False:#len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    
                    rospy.loginfo("[TRAIN] Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    self.face_encodings.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])#, known_face_locations=face_bounding_boxes)[0])
                    self.names.append(class_dir)

    def train_data(self):

        rospy.loginfo("[TRAIN] Starting the training phase")
        self.pub_speech.publish("Starting to train")

        # Determine how many neighbors to use for weighting in the KNN classifier
        if self.n_neighbors is None:
            self.n_neighbors = int(round(math.sqrt(len(self.face_encodings))))
            rospy.loginfo("[TRAIN] K was chosen automatically")
        else:
            rospy.loginfo("[TRAIN] K was provided")

        rospy.loginfo("[TRAIN] Its value is " + str(self.n_neighbors))
                          
        #self.n_neighbors = k

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.knn_algo, weights='distance')
        knn_clf.fit(self.face_encodings, self.names)

        # Save the trained KNN classifier
        if self.model_save_path is not None:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
        
        rospy.loginfo("[TRAIN] Training complete")
        self.pub_speech.publish("Training done")

    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
            if(self.msg_command.data == "memorize_person"):
                self.loopRate = rospy.Rate(30)
                self.msg_command.data = ""
                self.start_time = timeit.default_timer()

                # Routine for training, only necessary after all people have been added
                self.load_faces()
                self.train_data()

                self.end_time = timeit.default_timer()
                rospy.loginfo(str(self.end_time - self.start_time) + " sec.")
                self.pub_enable_face.publish(True)
                
if __name__ == "__main__":
    train = Trainer()
