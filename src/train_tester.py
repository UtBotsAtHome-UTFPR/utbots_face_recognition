import os
import train
import shutil
import random

class TrainTester:

    def __init__(self):
        # Should it crop the faces?
        # What should the value of k be?
        # How to name the trained data
        
        self.crop_faces = False

        self.k = 1

        self.people_amount = 0

        
        self.testing_faces_path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces/"
        self.all_faces_path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces_backup/"

        # DO NOT PUT ALL_FACES_PATH AS PARAMETERS HERE
        if os.path.exists(self.testing_faces_path):
            shutil.rmtree(self.testing_faces_path)

        self.load_images()

    
    
    # Function to load a random number (and random names) of people into the training set
    def load_images(self):
        if not os.path.exists(self.all_faces_path):
            return False
        
        os.mkdir(self.testing_faces_path)

        # Trained models path
        if not os.path.exists(os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/trained_models/"):
            os.mkdir(os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/trained_models/")

        all_people = os.listdir(self.all_faces_path)
        
        for n_people in range(1, len(all_people)):
            
            # For each possible number of images (POTENTIALLY RANDOMIZE WHICH PICTURES LATER)
            #for num_pics in range(15):
            if os.path.exists(self.testing_faces_path):
                shutil.rmtree(self.testing_faces_path)
                os.mkdir(self.testing_faces_path)

            people = []
            
            # Pode estar rerrodando código
            for person_number in range(n_people + 1):
                people.append(random.choice(all_people))
                all_people.remove(people[-1])

            for person in people:

                os.mkdir(self.testing_faces_path + person)
                for i in range(1 ,15):
                    for j in range(i):
                        # Take the picture inside path+"/person", crop it if needed, move it into training
                        if self.crop_faces == True:
                            pass
                        else:
                            # Load picture into faces/name
                            shutil.copyfile(self.all_faces_path + person + "/" + str(j) + ".jpg", self.testing_faces_path + person + "/" + str(j) + ".jpg")
                    
                    # Run the training routine
                    print(str(len(person)) + " people and " + str(i) + " images takes :")
                    train.Trainer("/usb_cam/image_raw", str(person_number) + "people_and" + str(i + 1) + "images_no_crop")

            
            all_people = os.listdir(self.all_faces_path)

        return True
    
    
    
    # Function to load different numbers of images into the named folder

if __name__ == "__main__":
    a = TrainTester()
    