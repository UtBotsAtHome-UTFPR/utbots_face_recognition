#!/usr/bin/python3

import os
import os.path
import face_recognition
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from voice_msgs.msg import NLU
import cv2
from cv_bridge import CvBridge
import shutil
import time
import smach

class SmPictureTaker(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["registered","aborted","wait_name"])
        self.picture_taker = PictureTaker(
            # "/camera/rgb/image_color"
            "/usb_cam/image_raw"
            )
        time.sleep(2)
        self.picture_taker.tts_publisher("I will give you instructions for face recognition ... Firstly, say your name", "Say your name ")
        
    def execute(self, userdata):
        # self.picture_taker.tts_publisher("I will give you instructions for face recognition ... Firstly, say your name", "Say your name ")
        state = self.picture_taker.capture_process()
        return state

class PictureTaker:
    def __init__(self, new_topic_rgbImg):

        self.train_dir = os.path.realpath(os.path.dirname(__file__)) + "/../faces"
        
        # OpenCV
        self.cv_img = None           # CvImage
        self.bridge = CvBridge()

        # Messages
        self.msg_command = String()
        self.msg_name = String()

        # Flags
        self.new_name = False

        # Publisher
        self.pub_instructions = rospy.Publisher("/utbots/voice/tts/robot_speech", String, queue_size=1)

        # Subscribers
        self.sub_rgbImg = rospy.Subscriber(new_topic_rgbImg, Image, self.callback_rgbImg)
        self.sub_is_talking = rospy.Subscriber("/utbots/voice/tts/is_robot_talking", Bool, self.callback_isTalking)
        self.sub_nlu_msg = rospy.Subscriber("/utbots/voice/nlu_msg", NLU, self.callback_nlumsg)

        # Subscriber variable 
        self.is_talking = False

        # ROS node
        # rospy.init_node('face_recognizer_new_person', anonymous=True)
        
        # Time
        self.loopRate = rospy.Rate(30)

        self.pic_quantity = 15

        # Eventually update verbose for making debugging easier (when it uses voice commands and so on)
        self.verbose = True # If set to True prints to the terminal when an image is unfit
        self.should_face_crop = True # If set to True, crops the images being taken to have only a face on them

        # Algorithm variables
        self.names = []
        self.face_encodings = []
        self.n_neighbors = None

    def callback_isTalking(self, msg):
        self.is_talking = msg.data

    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        self.cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.new_rgbImg = True
        
    def callback_nlumsg(self, msg):
        if(msg.database.data == "people"):
            self.msg_name.data = msg.text.data
            self.new_name = True

    # tts = text to speach
    def tts_publisher(self, speak, log="empty"):
        if(log != "empty"):
            rospy.loginfo("[REGISTER] " + log)
        else:
            rospy.loginfo("[REGISTER] " + speak)
        self.pub_instructions.publish(speak)    

    def picture_path_maker(self):

        name = self.msg_name.data
        path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces/" + name
        rospy.loginfo("[REGISTER] " + path)

        # Check whether the specified path exists or not
        if not os.path.exists(path):
            
            os.makedirs(path)
            
            self.tts_publisher("Your file is ready for training", "New person is ready for training")

        self.tts_publisher(f"Okay, {name}. Let's begin", "Created path for saving images")

        return path

    # Control for telling the user what to do for taking pictures
    def pic_instructions(self, i, speak):

        if i % 5 == 0:
            message = "Tilt your head directly into the camera"
        if i % 5 == 1:
            message = "Now upwards"
        if i % 5 == 2:
            message = "Downwards"
        if i % 5 == 3:
            message = "To the left"
        if i % 5 == 4:
            message = "To the right"

        if(speak):
            self.tts_publisher(message)

        time.sleep(1)

    def picture_taker(self, path):
        
        i = 0
        speak = True

        try: 
            while(i < self.pic_quantity):
                time.sleep(2)
                
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
                    
                    self.tts_publisher("Keep looking that way, i could not find you", "image has none or over 2 people")
                    
                    speak = False
                i += 1

            self.tts_publisher("You're done, congratulations and thank you very much", "Training complete")
            return "registered"
        except:
            rospy.logerr("Image not available")      
            return "aborted" 

    def capture_process(self):
        if(self.new_name == True):
            path = self.picture_path_maker()
            self.new_name = False
            state = self.picture_taker(path)
            return state
        else:
            return "wait_name"
