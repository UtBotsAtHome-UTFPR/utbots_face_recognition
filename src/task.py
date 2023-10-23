import rospy
from std_msgs.msg import String
import time
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import timeit
from darknet_ros_msgs.msg import BoundingBoxes
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import rospkg

class RecognitionTask:
    
    def __init__(self):

        self.bridge = CvBridge()

        self.msg_train_done = String
        self.msg_new_face_done = String
        self.gotImg = False
        self.gotMsg = False
        #self.msg_train_done.data = "no"
        #self.msg_new_face_done.data = "no"
        self.package_path = rospkg.RosPack().get_path('vision_tools')
    
        self.sub_enable = rospy.Subscriber("/utbots/vision/faces/train_done", String, self.callback_train_done)
        self.sub_enable = rospy.Subscriber("/utbots/vision/faces/new_face_done", String, self.callback_new_face_done)
        self.sub_rgbImg = rospy.Subscriber("/utbots/vision/faces/image", Image, self.callback_rgbImg)
        self.sub_is_done_talking = rospy.Subscriber("/is_robot_done_talking", String, self.callback_doneTalking)

        # Publishers
        self.pub_new_face_enable = rospy.Publisher("/utbots/vision/faces/new_face_enable", String, queue_size=1)
        self.pub_train_enable = rospy.Publisher("/utbots/vision/faces/train_enable", String, queue_size=1)
        self.pub_recognize_enable = rospy.Publisher("/utbots/vision/faces/recognize_enable", String, queue_size=1)
        self.pub_instructions = rospy.Publisher("/robot_speech", String, queue_size=1)

        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.callback_bBoxes)
        rospy.Subscriber("/darknet_ros/detection_image", Image, self.callback_image)

        # ROS node
        rospy.init_node('face_recognizer_task', anonymous=True)
        
        # Time
        self.loopRate = rospy.Rate(30)

    def callback_doneTalking(self, msg):
        self.done_talking = msg

    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        #self.cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.new_image = True

    def callback_train_done(self, msg):
        self.msg_train_done = msg
    
    def callback_new_face_done(self, msg):
        self.msg_new_face_done = msg

    def callback_bBoxes(self, msg):    
        self.detMsg = msg
        self.gotMsg = True

    def callback_image(self, msg):
        self.detImg = msg
        if self.gotMsg == True:
            self.gotImg = True

    def mainLoop(self):
        #while rospy.is_shutdown() == False:
        self.pub_instructions.publish("begin")
        self.pub_new_face_enable.publish("yes")
        time.sleep(1)
        self.pub_new_face_enable.publish("no")

        while self.msg_new_face_done.data != "yes":
            pass

        self.pub_instructions.publish("train")
        self.pub_train_enable.publish("yes")
        time.sleep(1)
        self.pub_train_enable.publish("no")

        start_time = timeit.default_timer()

        while self.msg_train_done.data != "yes":
            pass

        end_time = timeit.default_timer()

        self.pub_instructions.publish("Go to the crowd")
        while(end_time - start_time) < 60:
            end_time = timeit.default_timer()

        self.pub_instructions.publish("Recognizing")


        self.new_image = False
        self.pub_recognize_enable.publish("yes")
        time.sleep(1)

        #time.sleep(100) # Virar 180º
        
        time.sleep(8)

        cv_img = self.bridge.imgmsg_to_cv2(self.msg_rgbImg, desired_encoding="passthrough")
        
        #cv2.rectangle(cv_img, (10, 20), (60, 50), (0, 0, 255), 2)

        cv2.imwrite("recognized.jpeg", cv_img)

        # Take FIRST image and save it as pdf

        self.gotImg = False
        self.gotMsg = False

        while self.gotImg == False or self.gotMsg == False:
            pass

        # Create a PDF file
        c = canvas.Canvas(f"{self.package_path}/output.pdf", pagesize=letter)

        cv_image = self.bridge.imgmsg_to_cv2(self.detImg, desired_encoding="bgr8")
        # Save the image as PNG
        cv2.imwrite(f"{self.package_path}/output_image.png", cv_image)
        # Add the image to the PDF
        c.drawImage(f"{self.package_path}/output_image.png", 100, 500, width=400, height=300)

        textY0 = 480
        rospy.loginfo("Class | Probability")
        person_num = 0 
        c.drawString(100, textY0, "Class | Probability")
        for bbox in self.detMsg.bounding_boxes:
            if bbox.Class == "person":
                person_num += 1
            
        rospy.loginfo(f"Number of people | {person_num}")
        c.drawString(100, textY0, f"Number of people | {person_num}")
        # Save the image as PNG
        cv2.imwrite(f"{self.package_path}/face.png", cv_img)
        c.drawImage(f"{self.package_path}/face.png", 100, 0, width=400, height=300)

        # Save the PDF file
        c.save()

def main():
    a = RecognitionTask()
    a.mainLoop()






if __name__ == "__main__":
    main()