import rospy
from std_srvs.srv import Empty

print("Hello")
rospy.wait_for_service('/utbots_face_recognition/add_new_face')
print("boy")

try:
    # create a handle to the add_two_ints service
    add_new_face = rospy.ServiceProxy('/utbots_face_recognition/add_new_face', Empty)

    # simplified style
    resp1 = add_new_face()

    # formal style
    #resp2 = add_two_ints.call(AddTwoIntsRequest(x, y))

    #if not resp1.sum == (x + y):
    #    raise Exception("test failure, returned sum was %s"%resp1.sum)
    #if not resp2.sum == (x + y):
    #    raise Exception("test failure, returned sum was %s"%resp2.sum)
    #return resp1.sum
except rospy.ServiceException as e:
    print("Service call failed: %s"%e)

print("Done or killed")