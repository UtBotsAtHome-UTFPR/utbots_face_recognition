# Node for training the network for a new person

import math
from sklearn import neighbors
from std_srvs.srv import Empty
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import timeit
from std_msgs.msg import Bool

import actionlib
import utbots_actions.msg
from generic_msgs.msg import StringArray

class Trainer:

    def __init__(self, name, new_topic_rgbImg, save_name="trained_knn_model.clf", n: int=None):

        self.train_dir = os.path.realpath(os.path.dirname(__file__)) + "/../faces"
        self.model_save_path = os.path.realpath(os.path.dirname(__file__)) + "/../trained_models/" + save_name
        self.knn_algo = 'ball_tree'

        # OpenCV
        self.cv_img = None           # CvImage
        self.bridge = CvBridge()

        # Services
        self.train_service = rospy.Service('/utbots_face_recognition/train', Empty, self.train_srv)

        # Publishers
        self.pub_speech = rospy.Publisher("/robot_speech", String, queue_size=1)

        # ROS node
        rospy.init_node('face_recognizer_trainer', anonymous=True)
        
        # Time
        self.loopRate = rospy.Rate(1)

        # Eventually update verbose for making debugging easier (when it uses voice commands and so on)
        self.verbose = True # If set to True talks about it's actions

        # Algorithm variables
        self.names = []
        self.face_encodings = []
        self.n_neighbors = n

        self.goal = utbots_actions.msg.trainGoal()
        self.result = utbots_actions.msg.trainResult()
        self.feedback = utbots_actions.msg.trainFeedback()

        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, utbots_actions.msg.recognitionAction, execute_cb=self.train_action, auto_start = False)
        self._as.start()

        self.mainLoop()

    def train_action(self, goal):

        self.start_time = timeit.default_timer()

        if len(goal.names) > 0:
            self.load_faces()

        else:
            self.load_faces()

        self.result.model.data(self.train_data())

        self.end_time = timeit.default_timer()
        rospy.loginfo("Training took %i seconds", self.end_time - self.start_time)

    
    # Service that performs the training phase whenever called
    def train_srv(self, msg):
        self.start_time = timeit.default_timer()
        
        self.load_faces()
        self.train_data()

        self.end_time = timeit.default_timer()
        rospy.loginfo("Training took %i seconds", self.end_time - self.start_time)
        
        self.result.success.data = True

        self._as.set_succeeded(self.result)

    def load_faces(self, names = None):

        rospy.loginfo("Loading faces for training")
        self.pub_speech.publish("Loading faces for training")

        # Loop through each person in the training set
        if names == None:
            for class_dir in os.listdir(self.train_dir):
                if not os.path.isdir(os.path.join(self.train_dir, class_dir)):
                    continue
            
                # Loop through each training image for the current person
                for img_path in image_files_in_folder(os.path.join(self.train_dir, class_dir)):
                    image = face_recognition.load_image_file(img_path)

                    face_bounding_boxes = [(1, image.shape[1] - 1, image.shape[0] - 1, 1)]
                    
                    # Add face encoding for current image to the training set
                    self.face_encodings.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    self.names.append(class_dir)

    def train_data(self) -> str:

        rospy.loginfo("Starting the training phase")
        self.pub_speech.publish("Starting to train")

        # Determine how many neighbors to use for weighting in the KNN classifier
        if self.n_neighbors is None:
            self.n_neighbors = int(round(math.sqrt(len(self.face_encodings))))
            rospy.loginfo("K was chosen automatically")
        else:
            rospy.loginfo("K was provided")

        rospy.loginfo("It's value is " + str(self.n_neighbors))

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.knn_algo, weights='distance')
        knn_clf.fit(self.face_encodings, self.names)

        # Save the trained KNN classifier
        if self.model_save_path is not None:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)
        
        rospy.loginfo("Training complete")
        self.pub_speech.publish("Training done")

        return str(pickle.dumps(knn_clf))

    def mainLoop(self):
        # Just keeps the node running
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()

if __name__ == "__main__":

    train = Trainer(rospy.get_name(), "/usb_cam/image_raw")