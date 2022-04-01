import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import scipy.io as sio

""" Sources: 
    https://bioimagesuiteweb.github.io/webapp/mni2tal.html
    https://www.alivelearn.net/?p=1456
"""

cwd = os.path.dirname(os.path.abspath(__file__))+'\\'

def _spm_matrix(p):
    """Matrix transformation.
    Parameters
    ----------
    p : array_like
        Vector of floats for defining each tranformation. p must be a vector of
        length 9.
    Returns
    -------
    Pr : array_like
        The tranformed array.
    """
    q = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    p.extend(q[len(p):12])

    # Translation t :
    t = np.array([[1, 0, 0, p[0]],
                  [0, 1, 0, p[1]],
                  [0, 0, 1, p[2]],
                  [0, 0, 0, 1]])
    # Rotation 1 :
    r1 = np.array([[1, 0, 0, 0],
                   [0, np.cos(p[3]), np.sin(p[3]), 0],
                   [0, -np.sin(p[3]), np.cos(p[3]), 0],
                   [0, 0, 0, 1]])
    # Rotation 2 :
    r2 = np.array([[np.cos(p[4]), 0, np.sin(p[4]), 0],
                   [0, 1, 0, 0],
                   [-np.sin(p[4]), 0, np.cos(p[4]), 0],
                   [0, 0, 0, 1]])
    # Rotation 3 :
    r3 = np.array([[np.cos(p[5]), np.sin(p[5]), 0, 0],
                   [-np.sin(p[5]), np.cos(p[5]), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    # Translation z :
    z = np.array([[p[6], 0, 0, 0],
                  [0, p[7], 0, 0],
                  [0, 0, p[8], 0],
                  [0, 0, 0, 1]])
    # Translation s :
    s = np.array([[1, p[9], p[10], 0],
                  [0, 1, p[11], 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return np.linalg.multi_dot([t, r1, r2, r3, z, s])


def tal2mni(xyz):
    """Transform Talairach coordinates into MNI.
    Parameters
    ----------
    xyz : array_like
        Array of Talairach coordinates of shape (n_sources, 3)
    Returns
    -------
    xyz_r : array_like
        Array of MNI coordinates of shape (n_sources, 3)
    """
    # Check xyz to be (n_sources, 3) :
    if (xyz.ndim != 2) or (xyz.shape[1] != 3):
        raise ValueError("The shape of xyz must be (N, 3).")
    n_sources = xyz.shape[0]

    # Transformation matrices, different zooms above/below AC :
    rotn = np.linalg.inv(_spm_matrix([0., 0., 0., .05]))
    upz = np.linalg.inv(_spm_matrix([0., 0., 0., 0., 0., 0., .99, .97, .92]))
    downz = np.linalg.inv(_spm_matrix([0., 0., 0., 0., 0., 0., .99, .97, .84]))

    # Apply rotation and translation :
    xyz = np.dot(rotn, np.c_[xyz, np.ones((n_sources, ))].T)
    tmp = np.array(xyz)[2, :] < 0.
    xyz[:, tmp] = np.dot(downz, xyz[:, tmp])
    xyz[:, ~tmp] = np.dot(upz, xyz[:, ~tmp])
    return np.array(xyz[0:3, :].T)


def plot_coordinates(xyz, input_coordinate = None, colored = None, save_path = None):
    
    if colored:
        img_path = 'blend_xy.png'
    else:
        img_path = 'MNI_T1_1mm_stripped_xy.png'

    D2_img = plt.imread(fname=cwd+img_path)
    if img_path == 'MNI_T1_1mm_stripped_xy.png':
        D3_img = np.reshape(D2_img, [217,181,181])
    elif img_path == 'blend_xy.png':
        D3_img = np.reshape(D2_img, [217,181,181,3])
    
    if input_coordinate is 'tal':
        xyz = tal2mni(np.array([xyz]))[0]
    else:
        pass
    x,y,z = int(xyz[0]), int(xyz[1]), int(xyz[2])

    print(search_area(np.array([x,y,z])))
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f'MNI Coordinate\nx:{x} , y: {y} , z: {z}')
    
    # _,z,_
    z_e = z - 90 - 18
    ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
    ax1.plot([0,1])
    plt.imshow(D3_img[:,z_e,:],cmap='Blues',  interpolation='nearest',extent=[90,-90,-126,90])
    plt.scatter(x,y,color='r')
    plt.xlabel('x')
    plt.ylabel('y', rotation=0)
    
    # _,_,x
    x_e = x - 90
    ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
    ax2.plot([0,1])
    plt.imshow(np.rot90(D3_img[:,:,x_e]),cmap='Blues',  interpolation='nearest',extent=[90,-126,-72,108])
    plt.scatter(y,z,color='r')
    plt.xlabel('y')
    plt.ylabel('z', rotation=0)

    # y,_,_
    y_e = y + (90 + 36)
    ax3 = fig.add_subplot(gs[1, :]) # row 1, span all columns
    ax3.plot([0,1])
    plt.imshow(np.flip(np.rot90(D3_img[-y_e,:,:],k=2),axis=1),cmap='Blues',  interpolation='nearest',extent=[90,-90,-72,108])
    plt.scatter(x,z,color='r')
    plt.xlabel('x')
    plt.ylabel('z', rotation=0)   
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def read_mat(filename):
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return mat


def mni2cor(MNI_coords):
    """ 
    Converts MNI coordinates to atlas coordinates
    """
    MNI_coords = np.array([MNI_coords[0], MNI_coords[1], MNI_coords[2],1])
    T_ = np.array([[2, 0, 0, -92], [0, 2, 0, -128], [0, 0, 2, -74], [0, 0, 0, 1]])
    T_ = np.linalg.inv(T_).T
    return np.rint(np.dot(np.array([MNI_coords]), T_)).astype(np.int32)[0, :3]


def search_area(MNI_coords):
    index = mni2cor(MNI_coords)
    mat_data = read_mat(filename = cwd+'TDdatabase.mat')
    DB = mat_data['DB']
    areas = []
    for i in range(len(DB)):
        graylevel = DB[i].mnilist[index[0]-1, index[1]-1, index[2]-1]
        if graylevel == 0:
            areas.append('undefined')
            continue
        else:
            areas.append(DB[i].anatomy[graylevel-1])
    return areas


MNI_coords = np.array([2, 4, 60])
plot_coordinates(MNI_coords, input_coordinate = 'mni', colored = True)
