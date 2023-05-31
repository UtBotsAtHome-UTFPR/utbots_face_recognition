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
from cv_bridge import CvBridge
import timeit

class Trainer:

    def __init__(self, new_topic_rgbImg, save_name="trained_knn_model.clf"):

        self.train_dir = os.path.realpath(os.path.dirname(__file__)) + "/../faces"
        self.model_save_path = os.path.realpath(os.path.dirname(__file__)) + "/../trained_models/" + save_name
        self.knn_algo = 'ball_tree'

        # OpenCV
        self.cv_img = None           # CvImage
        self.bridge = CvBridge()

        # Subscriber
        self.sub_rgbImg = rospy.Subscriber(new_topic_rgbImg, Image, self.callback_rgbImg)

        # Publisher
        self.pub_current_img = rospy.Publisher("/utbots/vision/faces/image", Image, queue_size=1)

        # ROS node
        rospy.init_node('face_recognizer_trainer', anonymous=True)
        
        # Time
        self.loopRate = rospy.Rate(30)

        # Eventually update verbose for making debugging easier (when it uses voice commands and so on)
        self.verbose = True # If set to True prints to the terminal when an image is unfit
        self.should_face_crop = False # If set to true, crops the images being taken to have only a face on them

        # Algorithm variables
        self.names = []
        self.face_encodings = []
        self.n_neighbors = None

        self.start_time = timeit.default_timer()
        
        # Routine for training, only necessary after all people have been added
        self.load_faces()
        self.train_data()

        self.end_time = timeit.default_timer()

        print(str(self.end_time - self.start_time) + " sec.")


    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        self.cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.new_rgbImg = True


    def load_faces(self):

        # Loop through each person in the training set
        for class_dir in os.listdir(self.train_dir):
            if not os.path.isdir(os.path.join(self.train_dir, class_dir)):
                continue
        
            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(self.train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if self.verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    self.face_encodings.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    self.names.append(class_dir)


    def train_data(self):

        # Determine how many neighbors to use for weighting in the KNN classifier
        if self.n_neighbors is None:
            self.n_neighbors = int(round(math.sqrt(len(self.face_encodings))))
            if self.verbose:
                print("Chose n_neighbors automatically:", self.n_neighbors)

        #self.n_neighbors = k

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.knn_algo, weights='distance')
        knn_clf.fit(self.face_encodings, self.names)

        # Save the trained KNN classifier
        if self.model_save_path is not None:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)


if __name__ == "__main__":

    train = Trainer("/usb_cam/image_raw")
