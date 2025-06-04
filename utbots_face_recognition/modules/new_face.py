import os
import os.path
from deepface import DeepFace
import cv2

# Add capability to search for a person by walking around the room, or at least looking around

class PictureTaker:
    def __init__(self):

        # Should create /faces
        self.train_dir = os.path.realpath(os.path.dirname(__file__)) + "/../faces"


    def picture_path_maker(self, name="Operator"):
        """
        Creates and returns the path to save images when given an Operator name. If name is aready used deletes current pictures

        ## Parameters
        - **name**: str = "Operator"

        ## Return
        - **path**: str
        """

        path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces/" + name + "/"

        # Check whether the specified path exists or not
        if not os.path.exists(path):    
            os.makedirs(path)

        return path

    def crop_img(self, img: cv2.typing.MatLike, i=0):
        """
        Takes in image with single face, returns cropped image with just the face. If 0 or >1 face is detected returns False

        ## Parameters
        - **img**: MatLike

        ## Return
        - **cropped_img**  ||  **None**
        """
        try:
            detection = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=True)
        except:
            # If there are no faces return None
            return None

        # If only one face is found, make it a list
        if len(detection) != 1:
            return None
        # Step 2: Loop through detected faces

        region = detection[0]["region"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        
        # Step 3: Crop face from image
        face_img = img[y:y+h, x:x+w]

        # Optional: save cropped face (can help debug)
        #cv2.imwrite("face_.jpg", face_img)
        return face_img

        '''face_bounding_boxes = face_recognition.face_locations(img = img, model='knn')

        if len(face_bounding_boxes) == 1:
            color_adjusted_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Crop images to be only faces
            return color_adjusted_image[face_bounding_boxes[0][0]:face_bounding_boxes[0][2], face_bounding_boxes[0][3]:face_bounding_boxes[0][1]]
        
        return False'''
    
    def save_img(self, path, img):

        cv2.imwrite(path + ".jpeg", img)

if __name__ == "__main__":
    program = PictureTaker()

    img = cv2.imread('person.jpeg')

    img = program.crop_img(img)
    print(img)
    path = program.picture_path_maker("Operator")
    
    program.save_img(path + "Operator", img)