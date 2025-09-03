import numpy as np



t = np.load('trowel_pose_in_brick_frame.npy', allow_pickle=True)

print(t.shape)
print(t[0, :, :])

t1 = np.load('generated_pose_se3_K4.npy', allow_pickle=True)

print(t1.shape)
