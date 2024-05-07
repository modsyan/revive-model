import numpy as np
from sklearn.discriminant_analysis import StandardScaler


class FramePreprocessing:
    def __init__(self):
        self.num_repitation = 5
        self.num_channel = 3
        self.index_Spine_Base = 0
        self.index_Spine_Mid = 3
        self.index_Neck = 6
        self.index_Head = 9  # no orientation
        self.index_Shoulder_Left = 12
        self.index_Elbow_Left = 15
        self.index_Wrist_Left = 18
        self.index_Hand_Left = 21
        self.index_Shoulder_Right = 24
        self.index_Elbow_Right = 27
        self.index_Wrist_Right = 30
        self.index_Hand_Right = 33
        self.index_Hip_Left = 36
        self.index_Knee_Left = 39
        self.index_Ankle_Left = 42
        self.index_Foot_Left = 45  # no orientation
        self.index_Hip_Right = 48
        self.index_Knee_Right = 51
        self.index_Ankle_Right = 54
        self.index_Foot_Right = 57  # no orientation
        self.index_Spine_Shoulder = 60
        self.index_Tip_Left = 63  # no orientation
        self.index_Thumb_Left = 66  # no orientation
        self.index_Tip_Right = 69  # no orientation
        self.index_Thumb_Right = 72  # no orientation
        self.body_part = self.body_parts()
        self.num_timestep = 100
        self.num_joints = len(self.body_part)
        self.sc1 = StandardScaler()

    def body_parts(self):
        body_parts = [
            self.index_Spine_Base,
            self.index_Spine_Mid,
            self.index_Neck,
            self.index_Head,
            self.index_Shoulder_Left,
            self.index_Elbow_Left,
            self.index_Wrist_Left,
            self.index_Hand_Left,
            self.index_Shoulder_Right,
            self.index_Elbow_Right,
            self.index_Wrist_Right,
            self.index_Hand_Right,
            self.index_Hip_Left,
            self.index_Knee_Left,
            self.index_Ankle_Left,
            self.index_Foot_Left,
            self.index_Hip_Right,
            self.index_Knee_Right,
            self.index_Ankle_Right,
            self.index_Ankle_Right,
            self.index_Spine_Shoulder,
            self.index_Tip_Left,
            self.index_Thumb_Left,
            self.index_Tip_Right,
            self.index_Thumb_Right,
        ]
        return body_parts

    # def process(self, frame):
    #     X_frame = np.zeros((1, self.num_joints * self.num_channel)).astype("float32")
    #     counter = 0
    #     for parts in self.body_part:
    #         for i in range(self.num_channel):
    #             X_frame[0, counter + i] = frame[parts + i]
    #         counter += self.num_channel

    #     X_frame = self.sc1.transform(X_frame)

    #     return X_frame.reshape(1, self.num_timestep, self.num_joints, self.num_channel)

    def preprocessing(self, frame):
        X_frame = np.zeros((1, self.num_joints * self.num_channel)).astype("float32")
        counter = 0
        for parts in self.body_part:
            for i in range(self.num_channel):
                X_frame[0, counter + i] = frame[parts + i]
            counter += self.num_channel

        X_frame = self.sc1.transform(X_frame)

        return X_frame

    def reshape_frame(self, frame):
        X_frame = self.preprocessing(frame)
        return X_frame.reshape(1, self.num_timestep, self.num_joints, self.num_channel)


class FramePreprocessing:
    def __init__(self, data):
        self.num_repitation = 5
        self.num_channel = 3
        self.index_Spine_Base = 0
        self.index_Spine_Mid = 3
        self.index_Neck = 6
        self.index_Head = 9  # no orientation
        self.index_Shoulder_Left = 12
        self.index_Elbow_Left = 15
        self.index_Wrist_Left = 18
        self.index_Hand_Left = 21
        self.index_Shoulder_Right = 24
        self.index_Elbow_Right = 27
        self.index_Wrist_Right = 30
        self.index_Hand_Right = 33
        self.index_Hip_Left = 36
        self.index_Knee_Left = 39
        self.index_Ankle_Left = 42
        self.index_Foot_Left = 45  # no orientation
        self.index_Hip_Right = 48
        self.index_Knee_Right = 51
        self.index_Ankle_Right = 54
        self.index_Foot_Right = 57  # no orientation
        self.index_Spine_Shoulder = 60
        self.index_Tip_Left = 63  # no orientation
        self.index_Thumb_Left = 66  # no orientation
        self.index_Tip_Right = 69  # no orientation
        self.index_Thumb_Right = 72  # no orientation
        self.body_part = self.body_parts()
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 100
        self.new_label = []
        self.x = data  # Assuming the input data is already preprocessed
        self.batch_size = 1  # Real-time processing doesn't need batch processing
        self.num_joints = len(self.body_part)
        self.sc1 = StandardScaler()
        self.scaled_x = self.preprocessing(self.x)

    def body_parts(self):
        body_parts = [
            self.index_Spine_Base,
            self.index_Spine_Mid,
            self.index_Neck,
            self.index_Head,
            self.index_Shoulder_Left,
            self.index_Elbow_Left,
            self.index_Wrist_Left,
            self.index_Hand_Left,
            self.index_Shoulder_Right,
            self.index_Elbow_Right,
            self.index_Wrist_Right,
            self.index_Hand_Right,
            self.index_Hip_Left,
            self.index_Knee_Left,
            self.index_Ankle_Left,
            self.index_Foot_Left,
            self.index_Hip_Right,
            self.index_Knee_Right,
            self.index_Ankle_Right,
            self.index_Ankle_Right,
            self.index_Spine_Shoulder,
            self.index_Tip_Left,
            self.index_Thumb_Left,
            self.index_Tip_Right,
            self.index_Thumb_Right,
        ]
        return body_parts

    def preprocessing(self, frame):
        X_frame = np.zeros((1, self.num_joints * self.num_channel)).astype("float32")
        counter = 0
        for parts in self.body_part:
            for i in range(self.num_channel):
                X_frame[0, counter + i] = frame[parts + i]
            counter += self.num_channel

        # for row in range(self.x.shape[0]):
        #     counter = 0
        #     for parts in self.body_part:
        #         for i in range(self.num_channel):
        #             X_frame[row, counter + i] = self.x[row, parts + i]
        #         counter += self.num_channel 

        # Assuming sc1 is defined somewhere outside this class
        X_frame = self.sc1.transform(X_frame)

        # Reshape the frame for real-time processing
        X_frame = X_frame.reshape(
            1, self.num_timestep, self.num_joints, self.num_channel
        )

        return X_frame
