# Face Recognition

- ROS package that applies [face_recognition](https://pypi.org/project/face-recognition/)
- Tested Kinect V1 RGB and usb_cam
- For voice commands one must go to the [repository](https://github.com/UtBotsAtHome-UTFPR/utbots_voice?tab=readme-ov-file) and configure tts.

## Installation

### Building

```bash
cd catkin_ws/src
git clone https://github.com/UtBotsAtHome-UTFPR/utbots_face_recognition.git
cd ..
catkin_make
```

### Dependencies

This package depends on [freenect_launch](https://github.com/ros-drivers/freenect_stack) or [usb_cam](http://wiki.ros.org/usb_cam) and runs on python.

The code runs on Python 3.8 and you must use a virtualenv (Install with `pip install virtualenv`) with the path `/usr/bin/venv_utbots_face_recognition/bin/python` as the node expects its existence to run. Install the requirements:

```bash
cd /usr/bin
sudo python3 -m virtualenv venv_utbots_face_recognition --python=$(which python3)
roscd utbots_face_recognition/src
/usr/bin/venv_utbots_face_recognition/bin/python -m pip install -r requirements.txt
cd
sudo rosdep init
rosdep install --from-paths src --ignore-src -r -y
```

For utbots voice one must go to the [repository](https://github.com/UtBotsAtHome-UTFPR/utbots_voice?tab=readme-ov-file) and configure tts.

## Running

First, to run on usb:

```bash
roslaunch utbots_face_recognition usb_action_server.launch
```

To run on kinect:

```bash
roslaunch utbots_face_recognition freenect_action_server.launch
```

## Actions

```xml
new_face_action
train.action
recognition.action
```
