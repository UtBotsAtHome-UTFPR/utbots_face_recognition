# utbots_face_recognition basic guide

## Installation

Main dependencies:

```bash
pip install face-recognition
pip install numpy
pip install pickle
pip install opencv-python
pip install -U scikit-learn
sudo apt-get install python-catkin-tools python3-dev
```

```bash
cd ~/catkin_ws/src
git clone --recurse-submodules https://github.com/UtBotsAtHome-UTFPR/utbots_vision.git
git clone --recurse-submodules https://github.com/UtBotsAtHome-UTFPR/utbots_voice.git
```

For utbots voice one must go to the [repository](https://github.com/UtBotsAtHome-UTFPR/utbots_voice?tab=readme-ov-file) and configure tts.

Installing the package.

```bash
cd ~/catkin_ws/src
git clone https://github.com/UtBotsAtHome-UTFPR/utbots_face_recognition.git
cd ..
catkin_make
source devel/setup.bash
```

## Code guide for last minute changes

### Adding a new face

#### Adding multiple operator

One must go inside `picture_path_maker()` and comment the line that sets the operator name to be "operator" and change it for whatever may be of use. Alternatively, the `add_new_face` service can be changed to receive a string with the name as a parameter.

### Training phase

#### Changing k value

Update `self.n_neighbors` value to be the desired quantity. More will give better results to a certain point.

### Recognize

#### Constant recognition

Add the following code to the mainLoop.

```python
self.recognize()

if len(self.recognized_people.array) != 0: 
    self.pub_marked_people.publish(self.recognized_people)
    self.pub_marked_imgs.publish(self.bridge.cv2_to_imgmsg(cv2.cvtColor(self.pub_image, cv2.COLOR_BGR2RGB), encoding="passthrough"))
```

And ideally an enable flag to start and stop if needed.

#### Adding new people

In case there is the need for constant recognition alongside adding new people `load_train_data` must be called in the enable callback to load the training data.
