# Node for training the network for a new person

import math
from sklearn import neighbors
import os
import os.path
import pickle
#from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import rospy

class trainer:

    train_dir = os.getenv("HOME") + "/catkin_ws/src/face_recognition/faces"
    model_save_path = os.getenv("HOME") + "/catkin_ws/src/face_recognition/trained_knn_model.clf"
    n_neighbors = None
    knn_algo = 'ball_tree'
    verbose = False

    def __init__(self):
        """
        Trains a k-nearest neighbors classifier for face recognition.

        :param train_dir: directory that contains a sub-directory for each known person, with its name.

        (View in source code to see train_dir example tree structure)

        Structure:
            <train_dir>/
            ├── <person1>/
            │   ├── <somename1>.jpeg
            │   ├── <somename2>.jpeg
            │   ├── ...
            ├── <person2>/
            │   ├── <somename1>.jpeg
            │   └── <somename2>.jpeg
            └── ...

        :param model_save_path: (optional) path to save model on disk
        :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
        :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
        :param verbose: verbosity of training
        :return: returns knn classifier that was trained on the given data.
        """

        #rospy.init_node('train_face_recognition', anonymous=True)


        X = []
        y = []

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
                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if self.n_neighbors is None:
            self.n_neighbors = int(round(math.sqrt(len(X))))
            if self.verbose:
                print("Chose n_neighbors automatically:", self.n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if self.model_save_path is not None:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)


if __name__ == "__main__":

    train = trainer()
    #train.train()
