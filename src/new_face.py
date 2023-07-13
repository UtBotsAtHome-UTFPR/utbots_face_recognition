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
import rosnode

class PictureTaker:
    def __init__(self, new_topic_rgbImg):

        self.train_dir = os.path.realpath(os.path.dirname(__file__)) + "/../faces"
        
        # OpenCV
        self.cv_img = None           # CvImage
        self.bridge = CvBridge()

        # Publisher
        self.pub_instructions = rospy.Publisher("/robot_speech", String, queue_size=1)

        # Subscriber
        self.sub_rgbImg = rospy.Subscriber(new_topic_rgbImg, Image, self.callback_rgbImg)

        # ROS node
        rospy.init_node('face_recognizer_new_person', anonymous=True)
        
        self.pub_instructions.publish("I will give you instructions for face recognition by using my tiny winy little voice")
        # Time
        self.loopRate = rospy.Rate(30)

        self.pic_quantity = 15

        # Eventually update verbose for making debugging easier (when it uses voice commands and so on)
        self.verbose = True # If set to True prints to the terminal when an image is unfit
        self.should_face_crop = False # If set to true, crops the images being taken to have only a face on them

        # Algorithm variables
        self.names = []
        self.face_encodings = []
        self.n_neighbors = None

    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        self.cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.new_rgbImg = True

    def picture_path_maker(self):

        name = None
        path = None
        
        while(not name):
            self.pub_instructions.publish("I will give you instructions for face recognition by using my tiny winy little voice. uWu... Firstly, type in your name")
            rospy.loginfo("Type in your name: ")
            name = input()

            path = os.path.realpath(os.path.dirname(__file__)).rstrip("/src") + "/faces/" + name

            rospy.loginfo(path)
            # Check whether the specified path exists or not
            if not os.path.exists(path):
                
                os.makedirs(path)
                rospy.loginfo("New person is ready for training")
                self.pub_instructions("Your file is ready for training. Let's begin")

            else:
                rospy.loginfo("Do you want to replace the user by that name?: [Y/n] ")
                self.pub_instructions.publish("Unfortunately, it appears someone who goes by this name is already inside the database, would you like to override them? Type in y to confirm or n to choose another name")
                delete = input()
                if delete == 'y' or delete == 'Y':
                    try:
                        shutil.rmtree(path)
                        rospy.loginfo("directory was removed successfully")
                        self.pub_instructions.publish("Their directory was removed, let's begin")
                        os.makedirs(path)

                    # Ends function if directory can't be removed
                    except OSError as x:
                        rospy.logerr("Error occured: %s : %s" % (path, x.strerror))
                        return 

                else:
                    name = None
                    rospy.loginfo("Do you wish to exit ?: [Y/n] ")
                    self.pub_instructions.publish("In this case, do you wish to exit? Press y to exit or n to try again")
                    close = input()
                    if close == "y" or input == "Y":
                        exit(1)

        rospy.loginfo("Created path for saving images")
        rospy.loginfo("Ok. Let's begin")
        return path
    
    # Control for telling the user what to do for taking pictures
    def pic_instructions(self, i):

        # Turn all of these prints into tts


        if i % 5 == 0:
            print("Look directly into the camera")
        if i % 5 == 1:
            print("Look upwards")
        if i % 5 == 2:
            print("Look downwards")
        if i % 5 == 3:
            print("Look to the left of the camera")
        if i % 5 == 4:
            print("Look to the right of the camera")

        time.sleep(3)

    def picture_taker(self, path):
        
        i = 0
        while(i < self.pic_quantity):

            self.pic_instructions(i)

            # Take in pictures for the new person and save them
            img = self.cv_img
            face_bounding_boxes = face_recognition.face_locations(img)            

            if len(face_bounding_boxes) == 1:
                if self.should_face_crop:
                    # Weird coordinates because of top, left, bottom right order in bounding_boxes and top, bottom, left, right when cropping in cv2
                    cv2.imwrite(path + "/" + str(i) + '.jpg', cv2.cvtColor(img[face_bounding_boxes[0][0]:face_bounding_boxes[0][2], face_bounding_boxes[0][3]:face_bounding_boxes[0][1]], cv2.COLOR_BGR2RGB))
                else:
                    cv2.imwrite(path + "/" + str(i) + '.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            else:
                i -= 1
                print("Failed, ", end="")

            i += 1

        print("You're done, congratulations and thank you very much")

def main():
    
    program = PictureTaker("/usb_cam/image_raw")
    path = program.picture_path_maker()
    program.picture_taker(path)


if __name__ == "__main__":
    main()