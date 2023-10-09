import rospy
from std_msgs.msg import String
import time
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class RecognitionTask:
    
    def __init__(self):

        self.bridge = CvBridge()

        self.msg_train_done = String
        self.msg_new_face_done = String
        #self.msg_train_done.data = "no"
        #self.msg_new_face_done.data = "no"

    
        self.sub_enable = rospy.Subscriber("/utbots/vision/faces/train_done", String, self.callback_train_done)
        self.sub_enable = rospy.Subscriber("/utbots/vision/faces/new_face_done", String, self.callback_new_face_done)
        self.sub_rgbImg = rospy.Subscriber("/utbots/vision/faces/image", Image, self.callback_rgbImg)

        # Publishers
        self.pub_new_face_enable = rospy.Publisher("/utbots/vision/faces/new_face_enable", String, queue_size=1)
        self.pub_train_enable = rospy.Publisher("/utbots/vision/faces/train_enable", String, queue_size=1)
        self.pub_recognize_enable = rospy.Publisher("/utbots/vision/faces/recognize_enable", String, queue_size=1)

        # ROS node
        rospy.init_node('face_recognizer_task', anonymous=True)
        
        # Time
        self.loopRate = rospy.Rate(30)

    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        #self.cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.new_image = True

    def callback_train_done(self, msg):
        self.msg_train_done = msg
    
    def callback_new_face_done(self, msg):
        self.msg_new_face_done = msg

    def mainLoop(self):
        #while rospy.is_shutdown() == False:
        self.pub_new_face_enable.publish("yes")
        time.sleep(1)
        self.pub_new_face_enable.publish("no")

        while self.msg_new_face_done.data != "yes":
            pass

        self.pub_train_enable.publish("yes")
        time.sleep(1)
        self.pub_train_enable.publish("no")

        while self.msg_train_done.data != "yes":
            pass

        self.new_image = False
        self.pub_recognize_enable.publish("yes")
        time.sleep(1)
        time.sleep(100)
        while(not self.new_image):
            pass

        cv_img = self.bridge.imgmsg_to_cv2(self.msg_rgbImg, desired_encoding="passthrough")
        
        #cv2.rectangle(cv_img, (10, 20), (60, 50), (0, 0, 255), 2)

        cv2.imwrite("recognized.jpeg", cv_img)

        # Take FIRST image and save it as pdf

        


def main():
    a = RecognitionTask()
    a.mainLoop()






if __name__ == "__main__":
    main()