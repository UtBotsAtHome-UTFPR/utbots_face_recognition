#!/usr/bin/venv_utbots_face_recognition/bin/python
from std_msgs.msg import String

import os
import pickle
from cv_bridge import CvBridge
import cv2
import copy
import face_recognition
import base64

class Recognize_Action():
    """
    A module that makes offers image recognition features when given a trained knn model.

    ## Functions

    - **recognize(img, model, expected_faces)**

    Performs recognition, if no model is given it searches on the default model path
    """
    
    def __init__(self):
        
        # OpenCV
        self.cv_img = None
        self.bridge = CvBridge()

        # Flags
        self.new_rgbImg = False

        # Algorithm variables
        self.face_encodings = []

        self.load_train_data()


    def recognize(self, img:cv2.typing.MatLike, model: str=False, expected_faces: int=0):
        '''Performs recognition on an image.

        ## Parameters
        - **img**: MatLike

        Must be a cv2 type Image

        - **model**: str

        Model decoded into string (IDEALLY CHANGE IT SO IT'S ONLY THE MODEL NAME)

        - **expected_faces**: int

        Number of faces expected to be found, if 0 only checks for problems in the recognition

        ## Return
        - **tuple(MatLike, bbox:list) or None**

        If expected faces is correct or 0 returns the marked image + list of bounding boxes where each bounding box is a map
        
        '''

        if model:
            decoded_model = base64.b64decode(model)
            self.knn_clf = pickle.loads(decoded_model)
        else:
            self.load_train_data()

        self.draw_img = img.copy()

        print("[RECOGNIZE] Recognizing image")

        img, bbox = self.recognize_img(img)

        detect_count = sum(1 for name in bbox if name["id"] != "Unknown")

        if (detect_count != 0 and expected_faces == 0) or (detect_count == expected_faces and detect_count != 0): 
            return img, bbox
        
        else:
            print("[RECOGNIZE] Wrong number of faces detected in image")


    def load_train_data(self):

        file_directory = os.path.realpath(os.path.dirname(__file__)) + "/../models/model.clf"

        with open(file_directory, 'rb') as f:
            self.knn_clf = pickle.load(f)


    def draw_rec_on_faces(self, img, name, coordinates):

        top = coordinates[0]
        bottom = coordinates[1]
        left = coordinates[2]
        right = coordinates[3]

        # Draw a box around the face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        return img
        #pub_img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), encoding="passthrough")
        #self.pub_marked_imgs.publish(pub_img)



    def person_setter(self, i, is_match):
        ''' 
        Header header
        float64 probability
        int64 xmin
        int64 ymin
        int64 xmax
        int64 ymax
        string id
        string Class'''


        # Face locations saves the positions as: top, right, bottom, left
        bbox = {}
        
        bbox["xmin"] = self.face_locations[i][3]
        bbox["xmax"] = self.face_locations[i][1]
        bbox["ymin"] = self.face_locations[i][0]
        bbox["ymax"] = self.face_locations[i][2]

        bbox["Class"] = 'Person'
        bbox["id"] = self.knn_clf.predict(self.face_encodings)[i] if is_match else "Unknown"

        # Coordinates of the face in top, bottom, left, right order
        coordinates = [bbox["ymin"], bbox["ymax"], bbox["xmin"], bbox["xmax"]]

        self.draw_img = self.draw_rec_on_faces(self.draw_img, bbox["id"], coordinates)

        print("[RECOGNIZE] " + bbox["id"])
        
        return bbox


    def recognize_img(self, img):

        self.face_locations = face_recognition.face_locations(img)
        self.face_encodings = face_recognition.face_encodings(img, self.face_locations)

        if len(self.face_locations) == 0:
            return 

        # Calculates which person is more similar to each face
        closest_distances = self.knn_clf.kneighbors(self.face_encodings, n_neighbors=1)

        are_matches = [closest_distances[0][i][0] <= 0.3 for i in range(len(self.face_locations))] # Ver se da pra voltar pra 0.2 com várias fotos

        print("[RECOGNIZE] Recognized people are: ")
        # Adds each person in the image to recognized_people and alters img to show them
        bbox = []
        for i in range(len(are_matches)):
            bbox.append(self.person_setter(i, are_matches[i]))

        return self.draw_img, bbox


if __name__ == '__main__':
    classifier = Recognize_Action()

    

    image_path = 'pic.jpeg'
    image = cv2.imread(image_path)

    img, _ = classifier.recognize(image)

    cv2.imwrite("recognized.jpeg", img)