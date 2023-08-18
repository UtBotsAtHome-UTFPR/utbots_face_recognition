#!/usr/bin/python3

from new_face import SmPictureTaker
from train import SmTrainer
import smach
import rospy

def main():
    rospy.init_node('receptionist_sm')
    sm = smach.StateMachine(outcomes=['succeded', 'failed'])
    with sm:
        smach.StateMachine.add('TAKE_PICTURES', SmPictureTaker(), 
                               transitions={'registered':"TRAIN",
                                            'aborted':'failed'})
        smach.StateMachine.add('TRAIN', SmTrainer(), 
                               transitions={'trained':'succeded',
                                            'aborted':'failed'})
        
    outcome = sm.execute()
    
if __name__ == '__main__':
    main()