# AJUSTAR self.pic_quantity PARA A COMPETIÇÃO (NO NUC)

import os
import os.path
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import shutil
import time
from std_msgs.msg import Bool

# Add capability to search for a person by walking around the room, or at least looking around

class PictureTaker:
    def __init__(self, new_topic_rgbImg):

        self.train_dir = os.path.realpath(os.path.dirname(__file__)) + "/../faces"
        
        self.msg_enable = String()
        self.msg_enable.data = "no"

        # OpenCV
        self.cv_img = None           # CvImage
        self.bridge = CvBridge()

        # Publisher
        self.pub_instructions = rospy.Publisher("/robot_speech", String, queue_size=1)
        self.pub_done = rospy.Publisher("/utbots/vision/faces/new_face_done", String, queue_size=1)

        # Subscribers
        self.sub_rgbImg = rospy.Subscriber(new_topic_rgbImg, Image, self.callback_rgbImg)
        self.sub_is_done_talking = rospy.Subscriber("/is_robot_done_talking", String, self.callback_doneTalking)
        self.sub_enable = rospy.Subscriber("/utbots/vision/faces/new_face_enable", String, self.callback_enable)

        # Subscriber variable 
        self.done_talking = String("yes")

        # ROS node
        rospy.init_node('face_recognizer_new_person', anonymous=True)
        
        # Time
        self.loopRate = rospy.Rate(30)

        self.pic_quantity = 25

        # Eventually update verbose for making debugging easier (when it uses voice commands and so on)
        self.verbose = True # If set to True prints to the terminal when an image is unfit
        self.should_face_crop = True # If set to True, crops the images being taken to have only a face on them

        # Algorithm variables
        self.names = []
        self.face_encodings = []
        self.n_neighbors = None
        #done = Bool
        #done.data = False
        self.pub_done.publish("yes")

    def callback_enable(self, msg):
        self.msg_enable = msg
        if self.msg_enable.data == "yes":
            rospy.loginfo("[RECOGNIZE] Face Recognition New Face ENABLED")

        else:
            rospy.loginfo("[RECOGNIZE] Face Recognition New Face DISABLED")

    def callback_doneTalking(self, msg):
        self.done_talking = msg

    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        self.cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.new_rgbImg = True

    def picture_path_maker(self):

        name = None
        path = None
        
        while(not name):

            self.tts_publisher("Let's begin", "Let's begin")

            name = "Operator"#input()

            path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces/" + name

            rospy.loginfo(path)
            # Check whether the specified path exists or not
            if not os.path.exists(path):
                
                os.makedirs(path)
                
                #self.tts_publisher("Your file is ready for training", "New person is ready for training")

            else:
                #self.tts_publisher("Unfortunately, this name is taken, would you like to override it? Type in y to confirm or n to choose another name", "Do you want to replace the user by that name?: [Y/n] ")
                
                delete = 'y'#input()
                if delete == 'y' or delete == 'Y':
                    try:
                        shutil.rmtree(path)

                        #self.tts_publisher("Their directory was removed", "directory was removed successfully")

                        os.makedirs(path)

                    # Ends function if directory can't be removed
                    except OSError as x:
                        rospy.logerr("Error occured: %s : %s" % (path, x.strerror))
                        return 

                else:
                    name = None

                    #self.tts_publisher("In this case, do you wish to exit? Press y to exit or n to try again", "Do you wish to exit ?: [Y/n] ")

                    close = input()
                    if close == "y" or input == "Y":
                        exit(1)

        #self.tts_publisher("Ok. Let's begin", "Created path for saving images")

        return path
    
    # tts = text to speach
    def tts_publisher(self, speak, log="empty"):
        while(self.done_talking.data == "no"):
            pass

        if(log != "empty"):
            rospy.loginfo(log)
        else:
            rospy.loginfo(speak)
        self.pub_instructions.publish(speak)
        
        time.sleep(0.2)
        while(self.done_talking.data == "no"):
            pass

    # Control for telling the user what to do for taking pictures
    def pic_instructions(self, i, speak):

        if i % 5 == 0:
            message = "forward"
        if i % 5 == 1:
            message = "up"
        if i % 5 == 2:
            message = "down"
        if i % 5 == 3:
            message = "left"
        if i % 5 == 4:
            message = "right"

        if(speak):
            self.tts_publisher(message)

        time.sleep(1.5)

    def picture_taker(self, path):
        
        i = 0
        speak = True
        while(i < self.pic_quantity):
            
            self.pic_instructions(i, speak)

            # Take in pictures for the new person and save them
            img = self.cv_img
            face_bounding_boxes = face_recognition.face_locations(img)            

            if len(face_bounding_boxes) == 1:
                if self.should_face_crop:
                    # Weird coordinates because of top, left, bottom right order in bounding_boxes and top, bottom, left, right when cropping in cv2
                    cv2.imwrite(path + "/" + str(i) + '.jpg', cv2.cvtColor(img[face_bounding_boxes[0][0]:face_bounding_boxes[0][2], face_bounding_boxes[0][3]:face_bounding_boxes[0][1]], cv2.COLOR_BGR2RGB))
                else:
                    cv2.imwrite(path + "/" + str(i) + '.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                speak = True

            else:
                i -= 1
                
                self.tts_publisher("Again", "image has no one")
                
                speak = False
            i += 1

        #self.tts_publisher("You're done", "Necessary images are gathered")

    def mainLoop(self):
        while rospy.is_shutdown() == False:
            # Controls speed
            self.loopRate.sleep()
            # Put the enable
            if self.msg_enable.data == "yes":
                self.pub_done.publish("no")
                path = self.picture_path_maker()
                self.picture_taker(path)
            self.pub_done.publish("yes")


def execute():
    
    program = PictureTaker("/camera/rgb/image_color")
    program.mainLoop()
    


if __name__ == "__main__":
    execute()