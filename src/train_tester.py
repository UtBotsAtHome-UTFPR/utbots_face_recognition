import os
import train
import shutil
import random

class TrainTester:

    def __init__(self):
        # Should it crop the faces?
        # What should the value of k be?
        # How to name the trained data
        

        self.k = 1

        self.people_amount = 0

        
        self.testing_faces_path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces/"
        self.all_faces_path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces_backup/"

        # DO NOT PUT ALL_FACES_PATH IN THIS FUNCTION
        if os.path.exists(self.testing_faces_path):
            shutil.rmtree(self.testing_faces_path)

        self.load_images()

    
    
    # Function to load a random number (and random names) of people into the training set
    def load_images(self):
        if not os.path.exists(self.all_faces_path):
            return False
        
        all_people = os.listdir(self.all_faces_path)

        # For each possible number of images (POTENTIALLY RANDOMIZE LATER)
        for num_pics in range(15):

            for n_people in len(all_people):
                person = random.choice(all_people)
                all_people.remove(person)

                print(all_people, person)

                for i in range(num_pics):
                    # Take the picture inside path+"/person", crop it if needed, move it into training
                    pass

        return True
    
    
    
    # Function to load different numbers of images into the named folder

if __name__ == "__main__":
    a = TrainTester()
    