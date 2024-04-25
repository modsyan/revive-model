import os

kp_path = "../../dataset/Movements/Kinect/Positions/"
ka_path = "../../dataset/Movements/Kinect/Angles/"

kinect_positions = sorted([kp_path + f for f in os.listdir(kp_path) if f.endswith('.txt')])
kinect_angles = sorted([ka_path + f for f in os.listdir(ka_path) if f.endswith('.txt')])

print(kinect_positions[0])
print(kinect_angles[0])