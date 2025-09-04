import os
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R


def load_pose_npy(npy_path, sub_sample=1, trim_stationary=True,
                  vel_thresh=1e-3, T_total=None, dt=None):
    """
    Load a single demonstration from a .npy of shape (N, 4, 4) with homogeneous transforms.

    Args:
      npy_path        : path to .npy file (shape (N,4,4))
      sub_sample      : keep every k-th frame (default 1 = no subsampling)
      trim_stationary : if True, trim leading/trailing near-zero velocity segments
      vel_thresh      : threshold for trimming (L2 pos diff)
      T_total         : total duration (seconds) if you want to fix time span
      dt              : sampling period (seconds). If provided, used to build timestamps.
                        If neither T_total nor dt is given, defaults to dt=1.0s.

    Returns:
      p_raw : [ np.ndarray of shape (M,3) ]           (list with a single trajectory)
      q_raw : [ list of scipy Rotation objects of len M ]
      t_raw : [ np.ndarray of shape (M,) time stamps ]
      dt    : float (sampling period)
    """
    T_demo = np.load(npy_path)  # (N,4,4)
    assert T_demo.ndim == 3 and T_demo.shape[1:] == (4, 4), "expect (N,4,4) pose sequence"

    # Subsample if requested
    T_demo = T_demo[::sub_sample, :, :]
    N = T_demo.shape[0]

    # Extract positions and orientations
    p = T_demo[:, :3, 3]                          # (N,3)
    q = [R.from_matrix(T_demo[i, :3, :3]) for i in range(N)]  # list of Rotations

    # Optional trimming of leading/trailing stationary segments
    if trim_stationary and N >= 3:
        diffs = np.diff(p, axis=0)            # (N-1,3)
        vel_mag = np.linalg.norm(diffs, axis=1)
        first_idx = int(np.argmax(vel_mag > vel_thresh))
        last_idx  = int(len(vel_mag) - 1 - np.argmax(vel_mag[::-1] > vel_thresh))
        if first_idx < last_idx:
            # keep indices [first_idx : last_idx] inclusive on frames
            # note: vel_mag is length N-1, so frame indices are aligned
            p = p[first_idx:last_idx+1, :]
            q = q[first_idx:last_idx+1]
            N = p.shape[0]

    # Build time stamps + dt
    if dt is not None:
        t = np.arange(N, dtype=float) * dt
    elif T_total is not None:
        t = np.linspace(0.0, T_total, N, endpoint=False)
        dt = T_total / N
    else:
        # Fallback: assume 1 Hz (you can change this)
        dt = 1.0
        t = np.arange(N, dtype=float) * dt

    # Conform to your pipeline's expected structure: lists of trajectories
    p_raw = [p]            # list with one (M,3) array
    q_raw = [q]            # list with one list[Rotation]
    t_raw = [t]            # list with one (M,) array
    return p_raw, q_raw, t_raw, float(dt)



def _process_bag(path):
    """ Process .mat files that is converted from .bag files """

    data_ = loadmat(r"{}".format(path))
    data_ = data_['data_ee_pose']
    L = data_.shape[1]

    p_raw     = []
    q_raw     = []
    t_raw     = []

    sample_step = 5
    vel_thresh  = 1e-3 
    
    for l in range(L):
        data_l = data_[0, l]['pose'][0,0]
        pos_traj  = data_l[:3, ::sample_step]
        quat_traj = data_l[3:7, ::sample_step]
        time_traj = data_l[-1, ::sample_step].reshape(1,-1)

        raw_diff_pos = np.diff(pos_traj)
        vel_mag = np.linalg.norm(raw_diff_pos, axis=0).flatten()
        first_non_zero_index = np.argmax(vel_mag > vel_thresh)
        last_non_zero_index = len(vel_mag) - 1 - np.argmax(vel_mag[::-1] > vel_thresh)

        if first_non_zero_index >= last_non_zero_index:
            raise Exception("Sorry, vel are all zero")

        pos_traj  = pos_traj[:, first_non_zero_index:last_non_zero_index]
        quat_traj = quat_traj[:, first_non_zero_index:last_non_zero_index]
        time_traj = time_traj[:, first_non_zero_index:last_non_zero_index]
        
        p_raw.append(pos_traj.T)
        q_raw.append([R.from_quat(quat_traj[:, i]) for i in range(quat_traj.shape[1]) ])
        t_raw.append(time_traj.reshape(time_traj.shape[1]))

    dt = np.average([t_raw[0][i+1] - t_raw[0][i] for i in range(len(t_raw[0])-1)])

    return p_raw, q_raw, t_raw, dt




def _get_sequence(seq_file):
    """
    Returns a list of containing each line of `seq_file`
    as an element

    Args:
        seq_file (str): File with name of demonstration files
                        in each line

    Returns:
        [str]: List of demonstration files
    """
    seq = None
    with open(seq_file) as x:
        seq = [line.strip() for line in x]
    return seq




def load_clfd_dataset(task_id=1, num_traj=1, sub_sample=3):
    """
    Load data from clfd dataset

    Return:
    -------
        p_raw:  a LIST of L trajectories, each containing M observations of N dimension, or [M, N] ARRAY;
                M can vary and need not be same between trajectories

        q_raw:  a LIST of L trajectories, each containting a LIST of M (Scipy) Rotation objects;
                need to consistent with M from position
        
    Note:
    ----
        NO time stamp available in this dataset!

        [num_demos=9, trajectory_length=1000, data_dimension=7] 
        A data point consists of 7 elements: px,py,pz,qw,qx,qy,qz (3D position followed by quaternions in the scalar first format).
    """

    L = num_traj
    T = 10.0            # pick a time duration 

    file_path           = os.path.dirname(os.path.realpath(__file__))  
    dir_path            = os.path.dirname(file_path)
    data_path           = os.path.dirname(dir_path)

    seq_file    = os.path.join(data_path, "dataset", "clfd", "robottasks_pos_ori_sequence_4.txt")
    filenames   = _get_sequence(seq_file)
    datafile    = os.path.join(data_path, "dataset", "clfd", filenames[task_id])
    
    data        = np.load(datafile)[:, ::sub_sample, :]

    p_raw = []
    q_raw = []
    t_raw = []

    for l in range(L):
        M = data[l, :, :].shape[0]

        data_ori = np.zeros((M, 4))         # convert to scalar last format, consistent with Scipy convention
        w        = data[l, :, 3 ].copy()  
        xyz      = data[l, :, 4:].copy()
        data_ori[:, -1]  = w
        data_ori[:, 0:3] = xyz

        p_raw.append(data[l, :, :3])
        q_raw.append([R.from_quat(q) for q in data_ori.tolist()])
        t_raw.append(np.linspace(0, T, M, endpoint=False))   # hand engineer an equal-length time stamp

    dt = np.average([t_raw[0][i+1] - t_raw[0][i] for i in range(len(t_raw[0])-1)])

    return p_raw, q_raw, t_raw, dt




def load_demo_dataset():
    """
    Load demo data recorded from kniesthetic teaching
    """

    input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "demo", "all.mat")
    
    return _process_bag(input_path)




def load_UMI():

    traj = np.load("dataset/UMI/traj1.npy")

    q_raw = [R.from_matrix(traj[i, :3, :3]) for i in range(traj.shape[0])]

    p_raw = [traj[i, :3, -1] for i in range(traj.shape[0])]

    """provide dt"""
    # dt = 0.07

    """or provide T"""
    T = 5
    dt = T/traj.shape[0]

    t_raw = [dt*i for i in range(traj.shape[0])]

    return [np.vstack(p_raw)], [q_raw], [t_raw], dt