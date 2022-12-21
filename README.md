# Robotics - Lab 1 - Serial Manipulator
A program capable of making the Scorbot ER-7 draw a given image

## Authors
- Manuel Graça (96271)
- Tiago Lourinho (96327)
- Tiago Ferreira (96334)

### Code structure
- `main.py` - Program to run which contains the control panel
- `extract_points.py` - Functions related to image processing
- `utils.py` - Utility functions
- `adts/point.py` - Class representing a point in cartesian coordinates
- `adts/robot.py` - Class that handles robot manipulation
- `adts/writer.py` - Class that handles writing/reading to the serial port or text file
- `serial_test.py` - Test done to the serial port
- `simulator.py` - Serial Manipulator simulator

### Hyperparameters
- Inputs:
    - `IMAGE_PATH` (string): The image path in the computer
    - `SERIAL_PORT` (string): The serial port to use
- Debug information:
    - `WRITE_TO_SERIAL` (boolean): Whether to write to the serial port or to write the commands to a text file
    - `SHOW_CONTOURS_INFO` (boolean): Whether to show information about the contours found or not
- Contours filtering control:
    - `CONTOUR_MAX_ERROR` (float): The maximum error allowed when approximating the points in a contour
    - `JOIN_CONTOURS_THRESHOLD` (float): The percentage of the cropped image diagonal which is going to be used as the circle radius in contour post-processing
- Drawing control:
    - `ALLOW_LIFT_PEN` (boolean): Whether to allow the robot to lift the pen or not
    - `USE_ROLL` (boolean): Whether to use the roll to maximize pen stability or not
    - `DRAWING_AREA` (float): The drawing area in mm2
    - `ELEVATION` (float): The distance in mm the robot should lift the pen
    - `TIME_PER_POINT` (float): The time in ms to move between each point
    
### Usage

1. Open `main.py`

2. Change `IMAGE_PATH` according the desired image path and `SERIAL_PORT` to the used serial port

2. Set `WRITE_TO_SERIAL = False`, `SHOW_CONTOURS_INFO = True` and run the program to check if the simulated output matches expectations (if not, adjust the other hyperparameters accordingly) and change the control parameters to `True` and `False`, respectively

3. Position the robot gripper with the pen in the desired starting position on the paper

4. Run the program and wait for the setup time (loading points into the robot, which varies according to the number of points)

*Optional:* If any errors occur, restarting the robot and performing HOME should correct them.

*Note:* The logs and the commands sent can be seen in the directory `text_files/`

