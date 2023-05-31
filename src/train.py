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
import cv2
from cv_bridge import CvBridge
import shutil
import time
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

        # Algorithm variables

        # Eventually update verbose for making debugging easier (when it uses voice commands and so on)
        self.verbose = True # If set to True prints to the terminal when an image is unfit
        self.should_face_crop = False # If set to true, crops the images being taken to have only a face on them


        self.names = []
        self.face_encodings = []
        self.n_neighbors = None

        # Routine for taking and saving pictures of a subject
        #path = self.picture_path_maker()
        #self.picture_taker(path)

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


    def picture_path_maker(self):
        #Take pictures from ros and save them as files for training
        name = None
        path = None
        
        while(not name):
            name = input("Type in you name: ")

            path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces/" + name
            print(path)
            # Check whether the specified path exists or not
            if not os.path.exists(path):
                # Create a new directory because it does not exist
                os.makedirs(path)
                print("New person is ready for training")

            else:
                delete = input("Do you want to replace the user by that name?: [Y/n] ")
                if delete == 'y' or delete == 'Y':
                    try:
                        shutil.rmtree(path)
                        print("directory is removed successfully")
                        os.makedirs(path)

                    # Ends function if directory can't be removed
                    except OSError as x:
                        print("Error occured: %s : %s" % (path, x.strerror))
                        return 

                else:
                    name = None
        return path
    

    def picture_taker(self, path):
        
        i = 0
        while(i < 15):
            if i % 5 == 0:
                print("Look directly into the camera")
            if i % 5 == 1:
                print("Look upwards")
            if i % 5 == 2:
                print("Look downwards")
            if i % 5 == 3:
                print("Look to the left of the camera")
            if i % 5 == 4:
                print("Look to the right of the camera")

            time.sleep(3)

            # Take in pictures for the new person and save them
            img = self.cv_img
            face_bounding_boxes = face_recognition.face_locations(img)            

            if len(face_bounding_boxes) == 1:
                if self.should_face_crop:
                    cv2.imwrite(path + "/" + str(i) + '.jpg', cv2.cvtColor(img[face_bounding_boxes[0][0]:face_bounding_boxes[0][2], face_bounding_boxes[0][3]:face_bounding_boxes[0][1]], cv2.COLOR_BGR2RGB))
                else:
                    cv2.imwrite(path + "/" + str(i) + '.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            else:
                i -= 1
                print("Failed, ", end="")

            i += 1

        print("You're done, congratulations and thank you very much")
    

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
            
            if self.verbose:
                print("Images have been loaded")


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
    #train.train()
