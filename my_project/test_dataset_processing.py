from scipy import signal
import numpy as np


class TestProcessing():
    def __init__(self, data):
        self.num_repitation = 5
        self.num_channel = 3
        self.index_Spine_Base=0
        self.index_Spine_Mid=3
        self.index_Neck=6
        self.index_Head=9   # no orientation
        self.index_Shoulder_Left=12
        self.index_Elbow_Left=15
        self.index_Wrist_Left=18
        self.index_Hand_Left=21
        self.index_Shoulder_Right=24
        self.index_Elbow_Right=27
        self.index_Wrist_Right=30
        self.index_Hand_Right=33
        self.index_Hip_Left=36
        self.index_Knee_Left=39
        self.index_Ankle_Left=42
        self.index_Foot_Left=45   # no orientation    
        self.index_Hip_Right=48
        self.index_Knee_Right=51
        self.index_Ankle_Right=54
        self.index_Foot_Right=57   # no orientation
        self.index_Spine_Shoulder=60
        self.index_Tip_Left=63     # no orientation
        self.index_Thumb_Left=66   # no orientation
        self.index_Tip_Right=69    # no orientation
        self.index_Thumb_Right=72  # no orientation
        #self.dir = dir 
        self.body_part = self.body_parts()       
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 100
        self.new_label = []
        #self.x = self.import_dataset()
        self.x = data
        self.batch_size = int(self.x.shape[0]/100)
        self.num_joints = len(self.body_part)
        #self.sc1 = StandardScaler()
        #self.sc2 = StandardScaler()
        self.x = self.filter(self.x)
        self.scaled_x = self.preprocessing()
                
    def filter(self, data):
        """
        Parameters
        ----------
        data : numpy array
            2D numpy array.
    
        Returns
        -------
        data : numpy array
            returns the filters data.
        """
        for i in range(data.shape[1]):
            if data[0,i] == 1 or data[0,i] == 2:
                continue
            else:
                raf_x = data[:,i]
                b, a = signal.butter(3, (2/30), 'low')
                y = signal.filtfilt(b, a, raf_x)
                data[:,i] = y
        #return data[1:-1]
        return data

    def body_parts(self):
        body_parts = [self.index_Spine_Base, self.index_Spine_Mid, self.index_Neck, self.index_Head, self.index_Shoulder_Left, self.index_Elbow_Left, self.index_Wrist_Left, self.index_Hand_Left, self.index_Shoulder_Right, self.index_Elbow_Right, self.index_Wrist_Right, self.index_Hand_Right, self.index_Hip_Left, self.index_Knee_Left, self.index_Ankle_Left, self.index_Foot_Left, self.index_Hip_Right, self.index_Knee_Right, self.index_Ankle_Right, self.index_Ankle_Right, self.index_Spine_Shoulder, self.index_Tip_Left, self.index_Thumb_Left, self.index_Tip_Right, self.index_Thumb_Right
]
        return body_parts
    
    def preprocessing(self):
        X_train = np.zeros((1,self.num_joints*self.num_channel)).astype('float32')
        # for row in range(self.x.shape[0]):
        #     counter = 0
        #     for parts in self.body_part:
        #         for i in range(self.num_channel):
        #             X_train[row, counter+i] = self.x[row, parts+i]
        #         counter += self.num_channel 


        counter = 0
        for parts in self.body_part:
            for i in range(self.num_channel):
                X_train[0, counter+i] = self.x[0, parts+i]
            counter += self.num_channel 
        
        X_train = sc1.transform(X_train)         
                        
        X_train_ = np.zeros((self.batch_size, self.num_timestep, self.num_joints, self.num_channel))
        
        for batch in range(X_train_.shape[0]):
            for timestep in range(X_train_.shape[1]):
                for node in range(X_train_.shape[2]):
                    for channel in range(X_train_.shape[3]):
                        X_train_[batch,timestep,node,channel] = X_train[timestep+(batch*self.num_timestep),channel+(node*self.num_channel)]
                        
        X_train = X_train_
        return X_train