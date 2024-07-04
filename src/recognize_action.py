#! /usr/bin/env python
# Implementar o recognize action_server
# Implementar a bt
import rospy
import actionlib
#from utbots_face_recognition import msg as recognize_action
import utbots_actions.msg
from sensor_msgs.msg import Image, RegionOfInterest
from vision_msgs.msg import Object, ObjectArray
from std_msgs.msg import String

import os
import pickle
from cv_bridge import CvBridge
import cv2
import copy
import face_recognition

class Recognize_Action(object):

    # create messages that are used to publish feedback/result
    _result = utbots_actions.msg.recognitionResult
    
    def __init__(self, name, new_topic_rgbImg):
        
        # OpenCV
        self.cv_img = None           # CvImage
        self.bridge = CvBridge()     # Transformation from rosImage to cvImage

        # Flags
        self.new_rgbImg = False

        # Messages
        self.msg_enable = String()
        self.msg_enable.data = "no"

        # Subscriber
        self.sub_rgbImg = rospy.Subscriber(new_topic_rgbImg, Image, self.callback_rgbImg)

        # Publisher
        self.pub_marked_people = rospy.Publisher("/utbots/vision/faces/recognized", ObjectArray, queue_size=1)  # Bounding boxes for their faces with name
        self.pub_marked_imgs = rospy.Publisher("/utbots/vision/image/marked", Image, queue_size=1)

        # Publisher variables
        self.recognized_people = ObjectArray()
        self.marked_img = Image()

        # Algorithm variables
        self.face_encodings = []

        # Image that shows recognition
        self.edited_image = None
        self.pub_image = None

        self.load_train_data()

        # Declaring the action.
        ''' This is the last step of the init phase to avoid accessing undeclared variables. Alternatively _as can be declared sooner, however, _as.start must take memory safety into consideration'''
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, utbots_actions.msg.recognitionAction, execute_cb=self.recognize_action, auto_start = False)
        self._as.start()

        # Makes using 
        self.loopRate = rospy.Rate(1)

        self.mainLoop()

    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        self.cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.new_rgbImg = True

    def recognize_action(self, goal):
        # helper variables
        success = True
        '''
        Since this action is very short and not looping we won't check preemption.
        Basically: this could've been a service, however, considering it's use case it was decided to use an action
        '''

        # publish info to the console for the user
        rospy.loginfo("[RECOGNIZE] Loading knn trained data model")
        self.load_train_data()

        
        if self.new_rgbImg:
            rospy.loginfo("[RECOGNIZE] Recognizing image")
            self.recognize()

            if len(self.recognized_people.array) != 0: 
                self.pub_marked_people.publish(self.recognized_people)
                img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(self.pub_image, cv2.COLOR_BGR2RGB), encoding="passthrough")
                self.pub_marked_imgs.publish(img)
                action_res = utbots_actions.msg.recognitionResult()
                action_res.people.array = self.recognized_people.array
                #print(action_res.people.array)
                action_res.image.data = img.data
                action_res.success.data = True
                self._as.set_succeeded(action_res)
            
            else:
                rospy.loginfo("[RECOGNIZE] No faces detected in image")
                self._as.set_aborted()
        
        else:
            rospy.loginfo("[RECOGNIZE] No new image for recognition")
            self._as.set_aborted()

    def load_train_data(self):

        file_directory = os.path.realpath(os.path.dirname(__file__)) + "/../trained_models/trained_knn_model.clf"

        with open(file_directory, 'rb') as f:
            self.knn_clf = pickle.load(f)

    def draw_rec_on_faces(self, name, coordinates):
        img = self.edited_image

        top = coordinates[0]
        bottom = coordinates[1]
        left = coordinates[2]
        right = coordinates[3]

        # Draw a box around the face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    def person_setter(self, i, is_match):
        person = Object()

        # Face locations saves the positions as: top, right, bottom, left
        bbox = RegionOfInterest()
        bbox.x_offset = self.face_locations[i][3]
        bbox.y_offset = self.face_locations[i][0]
        bbox.height = self.face_locations[i][2] - self.face_locations[i][0]
        bbox.width = self.face_locations[i][1] - self.face_locations[i][3]

        person.roi = bbox
        
        person.category.data = "Person"

        person.id.data = self.knn_clf.predict(self.face_encodings)[i] if is_match else "Unknown"

        # Coordinates of the face in top, bottom, left, right order
        coordinates = [bbox.y_offset, bbox.y_offset + bbox.height, bbox.x_offset, bbox.x_offset + bbox.width]

        self.draw_rec_on_faces(person.id.data, coordinates)

        #person.parent_img.data = self.msg_rgbImg

        # Shows who has been found on the terminal window
        rospy.loginfo("[RECOGNIZE] " + person.id.data)
        
        return person

    def recognize(self):
        self.pub_image = self.cv_img

        self.face_locations = face_recognition.face_locations(self.cv_img)
        self.face_encodings = face_recognition.face_encodings(self.cv_img, self.face_locations)

        if len(self.face_locations) == 0:
            #self.pub_marked_imgs.publish(self.bridge.cv2_to_imgmsg(cv2.cvtColor(self.pub_image, cv2.COLOR_BGR2RGB), encoding="passthrough"))
            self.pub_marked_imgs.publish(self.bridge.cv2_to_imgmsg(self.pub_image, encoding="passthrough"))
            return 

        # Calculates which person is more similar to each face
        closest_distances = self.knn_clf.kneighbors(self.face_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(self.face_locations))]
  
        self.recognized_people.array.clear()

        rospy.loginfo("[RECOGNIZE] Recognized people are: ")
        # Adds each person in the image to recognized_people and alters img to show them
        self.edited_image = self.cv_img
        for i in range(len(are_matches)):
            self.recognized_people.array.append(self.person_setter(i, are_matches[i]))

        self.pub_image = self.edited_image

    def mainLoop(self):
        rospy.loginfo("[RECOGNIZE] Running recognition system")
        # Just keeps the node running
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
        
if __name__ == '__main__':
    rospy.init_node('recognition', anonymous=False)

    # Transformar em um parâmetro quando tivermos ROS 2 (O jeito em ROS 2 é diferente então não quis fazer agora)
    server = Recognize_Action(rospy.get_name(), 
        #"/camera/rgb/image_color")
        "/usb_cam/image_raw")
    rospy.spin()