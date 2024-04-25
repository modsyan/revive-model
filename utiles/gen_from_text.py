
from numpy import genfromtxt

pos_path = "../../dataset/Movements/Vicon/Positions/m03_s10_positions.txt"
ang_path = "../../dataset/Movements/Vicon/Angles/m03_s10_angles.txt"

print("num of frames")
print(genfromtxt(pos_path).shape[0])

incorrect_path = "../../dataset/Incorrect Segmented Movements/Vicon/Positions/m01_s02_e01_positions_inc.txt"

print("num of frames \"Incorrect\"")
print(genfromtxt(incorrect_path).shape[0])


print("Loading positions...")
print(genfromtxt(pos_path))

print("\n\n\n")
print("Loading angles")
print(genfromtxt(ang_path)[0])

