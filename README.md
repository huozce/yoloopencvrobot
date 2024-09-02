# yoloopencvrobot
I trained a YOLO model to recognize apples, focusing mostly on green and yellow ones. The model effectively detects and tracks apples, sending the detected coordinates via serial communication for motor movements. The setup includes two servos for rotating the robot and one for shooting Nerf turrets. While the model works well for tracking apples and rotating the robot accordingly, there is a delay when it comes to shooting and moving the robot, which affects performance.

If you're only interested in tracking the apple and rotating the robot, the setup works just fine. To improve it, you can remove the third servo that controls shooting and delete the delays in the code.

The "train29" folder contains the trained model that works well for this project. In the "computervisionforapple.py" file, we use the YOLO model to detect apples and send the coordinates of the detected apples for the project’s purposes.

In the "arduinocode" folder, there are some delays related to the third servo, which causes the robot to not operate simultaneously. The third servo's pushing mechanism is poorly implemented in the code; it moves only after receiving the coordinates of the detected object, then waits before triggering the servo, leading to delays. This issue could be fixed by using millis() in Arduino instead of delays.

If you want to train the model yourself, there are images and labels available in the "yoloimagesifyouneed" folder.





