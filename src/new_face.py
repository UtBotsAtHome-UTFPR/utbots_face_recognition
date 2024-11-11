#!/usr/bin/venv_utbots_face_recognition/bin/python
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
import actionlib

import utbots_actions.msg

# Add capability to search for a person by walking around the room, or at least looking around

class PictureTaker:
    def __init__(self):

        self.train_dir = os.path.realpath(os.path.dirname(__file__)) + "/../faces"

        # OpenCV
        self.cv_img = None           # CvImage
        self.bridge = CvBridge()

        # Publisher
        self.pub_instructions = rospy.Publisher("/utbots/voice/tts/robot_speech", String, queue_size=1)

        # Subscribers

        try:
            img_topic = rospy.get_param("image_topic")
        except:
            img_topic = "/camera/rgb/image_color"

        self.sub_rgbImg = rospy.Subscriber(img_topic, Image, self.callback_rgbImg)
        self.sub_is_done_talking = rospy.Subscriber("/is_robot_done_talking", String, self.callback_doneTalking)

        # Subscriber variable 
        self.done_talking = String("yes")

        # ROS node
        rospy.init_node('new_face', anonymous=True)

        # Action
        self.goal = utbots_actions.msg.new_faceGoal()
        self.result = utbots_actions.msg.new_faceResult()
        self.feedback = utbots_actions.msg.new_faceFeedback()

        self._as = actionlib.SimpleActionServer('new_face', utbots_actions.msg.new_faceAction, execute_cb=self.new_face_action, auto_start = False)
        self._as.start()
        
        # Time
        self.loopRate = rospy.Rate(5)

        self.pic_quantity = 25


    def callback_doneTalking(self, msg):
        self.done_talking = msg

    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        self.cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.new_rgbImg = True

    def picture_path_maker(self, name="Operator"):
        
        rospy.loginfo("[NEW_FACE] Creating path for pictures")

        name = None
        path = None
        
        while(not name):

            self.tts_publisher("Let's begin, please move around a little while keeping your face towards the camera", "Let's begin")

            # Make it a parameter for rosservice to add many users
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
            
            rospy.loginfo("[NEW_FACE] Path created")
           

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
    def pic_instructions(self, i):

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

        #self.tts_publisher(message)

        time.sleep(1.5)

    def picture_taker(self, path):
        
        i = 0
        while(i < self.pic_quantity):

            self.success = True
            if self._as.is_preempt_requested():
                rospy.loginfo("[NEW_FACE] Action preempted")
                self._as.set_preempted()
                self.success = False
                break
            
            self.pic_instructions(i)

            # Take in pictures for the new person and save them
            img = self.cv_img
            face_bounding_boxes = face_recognition.face_locations(img)            

            # The library can crop the images during the training phase but this is about 20 to 30% faster
            if len(face_bounding_boxes) == 1:
                # Weird coordinates because of top, left, bottom right order in bounding_boxes and top, bottom, left, right when cropping in cv2
                cv2.imwrite(path + "/" + str(i) + '.jpg', cv2.cvtColor(img[face_bounding_boxes[0][0]:face_bounding_boxes[0][2], face_bounding_boxes[0][3]:face_bounding_boxes[0][1]], cv2.COLOR_BGR2RGB))

            else:
                i -= 1
                #self.tts_publisher("Again", "image has too many or too few people")

            self.feedback.pics_taken.data = i + 1
            self.feedback.image.data = self.msg_rgbImg.data

            self._as.publish_feedback(self.feedback)

            i += 1

        #self.tts_publisher("You're done", "Necessary images are gathered")
    
    def new_face_action(self, goal):

        self.success = False

        path = self.picture_path_maker(goal.name)
        
        self.pic_quantity = goal.n_pictures.data

        self.picture_taker(path)

        if self.success:
            self._as.set_succeeded(self.result)
        
        else:
            self._as.set_aborted()

    # Just keeps the node running
    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()

def execute():
    
    program = PictureTaker()
    program.mainLoop()

if __name__ == "__main__":
    execute()
