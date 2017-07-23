import time
import vrep
import numpy as np
import os
from PIL import Image

vrep.simxFinish(-1)
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

err = vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
vrep.simxSynchronous(clientID, True)

"""
inputInts and inputFloats will pass setting parameters into the Lua script
Note that the indexing in Lua starts with 1 unlike Python
"""

""" PARAMETERS """

# Mode: 0 for data generating, 1 for control
mode = 0

# [0] Data generating mode parameters
path_steps = 16
epochs = 20
path_type = "ik"
save_images = True
dataset_path = "../datasets/"
dir_name = "3dof-arm-test/"
save_path = dataset_path + dir_name + "images/"
textfile_name = "joint_state.txt"
textfile_path = dataset_path + dir_name + textfile_name
image_counter = 1
image_shape = (128, 128, 3)

# [1] Control mode parameters

# Shared parameters
num_joints = 3

# Create dataset save directory
if save_images and not os.path.isdir(dataset_path + dir_name):
    os.mkdir(dataset_path + dir_name)
    os.mkdir(save_path)


def follow_path(path, joint_handles, vs_handle, distance_handle, textfile=None):
    num_joints = len(joint_handles)
    steps = len(path) / num_joints
    global image_counter

    for s in range(steps):
        text_str = str(image_counter)
        for j in range(num_joints):
            err = vrep.simxSetJointPosition(clientID, joint_handles[j], path[s * num_joints + j], vrep.simx_opmode_oneshot)
            err, joint_pos = vrep.simxGetJointPosition(clientID, joint_handles[j], vrep.simx_opmode_blocking)
            joint_pos = np.round(joint_pos, decimals=5)
            text_str += " " + str(joint_pos)
        err, distance = vrep.simxReadDistance(clientID, distance_handle, vrep.simx_opmode_blocking)
        text_str += " " + str(np.round(distance, decimals=5))

        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)

        if save_images:
            err, resolution, image = vrep.simxGetVisionSensorImage(clientID, vs_handle, 0, vrep.simx_opmode_blocking)
            image = np.flip(np.array(image, dtype=np.uint8).reshape(image_shape), axis=0)
            image = Image.fromarray(image)

            image.save(save_path + "image" + str(image_counter) + ".jpg")
            image_counter += 1

            textfile.write(text_str + "\n")

    return


def generate_ikpath():
    inputInts = []
    inputFloats = []
    inputStrings = []
    inputBuffer = bytearray()
    err, _, ikpath, _, _ = vrep.simxCallScriptFunction(clientID, 'floor',
                                                              vrep.sim_scripttype_childscript,
                                                              'generate_ikpath', inputInts,
                                                              inputFloats, inputStrings,
                                                              inputBuffer, vrep.simx_opmode_blocking)
    return err, ikpath


def get_end_config(dist_threshold=0.3, max_search_time=1000):
    inputInts = []
    inputFloats = [dist_threshold, max_search_time]
    inputStrings = []
    inputBuffer = bytearray()
    err, _, end_config, _, _ = vrep.simxCallScriptFunction(clientID, 'floor',
                                                           vrep.sim_scripttype_childscript,
                                                           'get_end_config', inputInts,
                                                           inputFloats, inputStrings,
                                                           inputBuffer, vrep.simx_opmode_blocking)
    if len(end_config) == 0:
        end_config = get_end_config(dist_threshold=dist_threshold+0.1)

    return end_config


def follow_interpolate_path(current_config, end_config, joint_handles):
    difference = np.array(end_config) - np.array(current_config)
    print "Difference", difference
    for j in range(num_joints):
        if difference[j] < -np.pi:
            difference[j] = difference[j] + 2 * np.pi
        elif difference[j] > np.pi:
            difference[j] = difference[j] - 2 * np.pi

    step_difference = difference / float(path_steps)

    for e in range(epochs):
        for j in range(num_joints):
            err, joint_pos = vrep.simxGetJointPosition(clientID, joint_handles[j], vrep.simx_opmode_blocking)
            err = vrep.simxSetJointPosition(clientID, joint_handles[j], joint_pos + step_difference[j], vrep.simx_opmode_oneshot)
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)

    return



inputInts = [mode, path_steps]
inputFloats = []
inputStrings = []
inputBuffer = bytearray()
err, _, _, _, _ = vrep.simxCallScriptFunction(clientID, 'floor',
                                              vrep.sim_scripttype_childscript,
                                              'python_init', inputInts,
                                              inputFloats, inputStrings,
                                              inputBuffer, vrep.simx_opmode_blocking)

# Get cube handle and position
err, cube = vrep.simxGetObjectHandle(clientID, "cube", vrep.simx_opmode_blocking)
print cube
err, init_cube_pos = vrep.simxGetObjectPosition(clientID, cube, -1, vrep.simx_opmode_blocking)

# Get joint handles
joint_handles = [-1, -1, -1]
for i in range(3):
    err, jh = vrep.simxGetObjectHandle(clientID, "joint" + str(i + 1), vrep.simx_opmode_blocking)
    joint_handles[i] = jh

# Get vision sensor handle
err, vs_diag = vrep.simxGetObjectHandle(clientID, "vs_diag", vrep.simx_opmode_oneshot_wait)

# Get distance handle
err, distance_handle = vrep.simxGetDistanceHandle(clientID, "tip_to_target", vrep.simx_opmode_blocking)

# Open textfile
textfile = open(textfile_path, "w")
err_count = 0
success_count = 0
while success_count < epochs:
    print("Epoch: {}/{}".format(success_count + 1, epochs))

    # Reset cube pos
    cube_x_pos = np.random.choice([-1.0, 1.0]) * np.random.uniform(0.1, 0.3)
    # cube_y_pos = np.random.choice([-1.0, 1.0]) * np.random.uniform(0.1, 0.4)
    cube_y_pos = np.random.uniform(0.1, 0.5)
    cube_z_pos = init_cube_pos[2]
    vrep.simxSetObjectPosition(clientID, cube, -1, [cube_x_pos, cube_y_pos, cube_z_pos], vrep.simx_opmode_oneshot)

    # Reset joint pos
    # jh[0]/joint1, range: [-1/2*pi, 1/2*pi]
    # jh[1]/joint2, range: [-3/4*pi, 0]
    # jh[2]/joint3, range: [-pi, 0]
    joint1_pos = np.random.uniform(0.25 * -np.pi, 0.25 * np.pi)
    joint2_pos = np.random.uniform(0.5 * -np.pi, 0)
    joint3_pos = np.random.uniform(0.5 * -np.pi, 0)
    current_config = [joint1_pos, joint2_pos, joint3_pos]
    vrep.simxSetJointPosition(clientID, joint_handles[0], joint1_pos, vrep.simx_opmode_oneshot)
    vrep.simxSetJointPosition(clientID, joint_handles[1], joint2_pos, vrep.simx_opmode_oneshot)
    vrep.simxSetJointPosition(clientID, joint_handles[2], joint3_pos, vrep.simx_opmode_oneshot)

    if path_type=='ik':
        err, ikpath = generate_ikpath()
        if err != 0:
            err_count += 1
            print("Non successful pathfinding, redoing epoch...")
        else:
            success_count += 1
        follow_path(ikpath, joint_handles, vs_diag, distance_handle, textfile)
    elif path_type=='interpolate':
        end_config = get_end_config()
        follow_interpolate_path(current_config, end_config, joint_handles)
    else:
        print "Not a valid path type"

# Close textfile
textfile.close()


print("Error count:", err_count)


err = vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)

