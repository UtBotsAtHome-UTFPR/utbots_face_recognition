import os
from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition
import math
import pickle
from sklearn import neighbors
import json

class train_subprocess:

    def __init__(self):
        f = open("../subprocess_communication/train_goal.json")

        data = json.load(f)

        self.train_dir = data["train_dir"]
        self.model_save_path = data["model_save_path"]
        self.train_result = data["train_result_dir"]
        
        self.face_encodings = []
        self.names = []

    def load_faces(self):

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

    def train_data(self):

        # Determine how many neighbors to use for weighting in the KNN classifier
    
        n_neighbors = int(round(math.sqrt(len(self.face_encodings))))

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
        knn_clf.fit(self.face_encodings, self.names)

        self.face_encodings = []
        self.names = []

        # Save the trained KNN classifier
        if self.model_save_path is not None:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        with open(self.train_result, 'w') as f:
            f.write(str("success"))
        
if __name__ == "__main__":
    object = train_subprocess()
    object.load_faces()
    object.train_data()