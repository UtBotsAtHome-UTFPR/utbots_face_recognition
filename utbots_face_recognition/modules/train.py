#!/usr/bin/venv_utbots_face_recognition/bin/python
import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import timeit
from std_msgs.msg import Bool
import subprocess
import sys
import base64
import time


class Trainer:

    """
    A module that makes offers training for facial recognition from images in a file.

    ## Functions

    - **train() -> bool**

    Performs training
    """

    def __init__(self, save_name="model.clf"):

        self.train_dir = os.path.realpath(os.path.dirname(__file__)) + "/../faces"
        self.model_save_path = os.path.realpath(os.path.dirname(__file__)) + "/../models/" + save_name

        # OpenCV
        self.bridge = CvBridge()

        # Algorithm variables
        self.process = None

    # Remember to check if there is a train subprocess already running
    def train(self) -> bool:
        """
        Starts the training procedure, returns False if there is one already running

        ## Return
        - **success**: bool
        """

        self.start_time = timeit.default_timer()

        # esse path terá que ser trocado (share directory sei lá o que), mas deve funcionar assim
        script_name = os.path.realpath(os.path.dirname(__file__)) + "/train_subprocess.py"  

        if self.process != None:
            print("Process already running, can't start new ones")
            return False
        
        self.process = subprocess.Popen([sys.executable, script_name, self.train_dir, self.model_save_path])
        
        return True
        #while self.process.poll() is None:
        #    pass
        '''    if self._as.is_preempt_requested():
                rospy.loginfo("[TRAIN] Action preempted")
                process.kill()
                self._as.set_preempted()
                self.success = False
                return
            self.loopRate.sleep()'''

        # Load the model as a string and set it as the result
        #self.result.model.data(model)

        

        print("Training complete")


    def poll(self, preempt=False):
        """
        Asks the process if model is done training, allows for preemption.

        Returns False if there is no process or it crashed, None if process is not donw and True if preemption succeeded

        ## Parameters
        - **preempt**: bool = False

        ## Return
        - **status**: bool | None
        """
        if self.process is None:
            return False

        if preempt:            
            self.process.kill()
            return True

        if self.process.poll() is None:
            return
        elif self.process.poll() == 1:
            print("[TRAIN] train_subprocess.py crashed")
            return False
        elif self.process.poll() == 2:
            print("[TRAIN] train_subprocess.py is badly named, path is wrong or does not exist")
            #self._as.set_aborted()
            return False
        
        self.end_time = timeit.default_timer()
        print(f"Training took {self.end_time - self.start_time} seconds")

        with open(self.model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)

        serialized_model = pickle.dumps(knn_clf)

        encoded_model = base64.b64encode(serialized_model).decode('utf-8')

        return encoded_model

        #self.result.model.data = encoded_model


if __name__ == "__main__":

    train = Trainer()

    train.train()

    while(True):
        poll = train.poll()

        if poll or poll == False:
            break
        