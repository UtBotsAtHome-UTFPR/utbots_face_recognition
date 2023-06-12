import os
import train
import shutil
import random
import timeit
import csv

class TrainTester:

    def __init__(self):
        # Should it crop the faces?
        # What should the value of k be?
        # How to name the trained data
        
        self.crop_faces = False

        self.k = 1

        self.people_amount = 0

        self.people_used = []
        self.images_used = []
        self.duration_used = []
        self.neighbors_used = []

        self.testing_faces_path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces/"
        self.all_faces_path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces_backup/"

        # DO NOT PUT ALL_FACES_PATH AS PARAMETERS HERE
        if os.path.exists(self.testing_faces_path):
            shutil.rmtree(self.testing_faces_path)

        self.load_images()

        with open('profiles1.csv', 'w', newline='') as file:
            self.writer = csv.writer(file)
            fields = ["people", "images", "n_neighbors", "duration"]
            self.writer.writerow(fields)
            for i in range(len(self.people_used)):
                self.writer.writerow([str(self.people_used[i]), str(self.images_used[i]), str(self.neighbors_used[i]), str(self.duration_used[i])])


    
    
    # Function to load a random number (and random names) of people into the training set
    def load_images(self):
        if not os.path.exists(self.all_faces_path):
            return False
        
        os.mkdir(self.testing_faces_path)

        # Trained models path
        if not os.path.exists(os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/trained_models/"):
            os.mkdir(os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/trained_models/")

        all_people = os.listdir(self.all_faces_path)
        print(all_people)

        people = []
        for n_people in range(1, len(all_people) + 1):
            people.append(random.choice(all_people))
            all_people.remove(people[-1])
            for n_imgs in range(1, len(os.listdir(self.all_faces_path + people[-1])) + 1):
                
                # Clear files
                if os.path.exists(self.testing_faces_path):
                    shutil.rmtree(self.testing_faces_path)
                    os.mkdir(self.testing_faces_path)
                
                # Create people
                for person in people:
                    os.mkdir(self.testing_faces_path + person)

                    # Copy images to people
                    for i in range(0, n_imgs):
                        shutil.copyfile(self.all_faces_path + person + "/" + str(i) + ".jpg", self.testing_faces_path + person + "/" + str(i) + ".jpg")
                        

                print()

                

                print(str(len(people)) + " people and " + str(n_imgs) + " images takes :\t\t" + str(len(people)))
                

                for n in range(1, 10):

                    # Does the training without cropping images
                    self.start_time = timeit.default_timer()

                    save_path = str(len(people)) + "people_and" + str(n_imgs) + "images" + str(n) + "neighbors" + "no_crop" 
                    train.Trainer("/usb_cam/image_raw", save_path, n, False)
                    
                    duration = timeit.default_timer() - self.start_time

                    self.people_used.append(len(people))
                    self.images_used.append(n_imgs)
                    self.duration_used.append(duration)
                    self.neighbors_used.append(n)


        return True
    
def add_to_csv():
    pass
    
    
    # Function to load different numbers of images into the named folder

if __name__ == "__main__":
    a = TrainTester()
    