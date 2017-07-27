import time
import vrep
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from image_utils import display

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
mode = 1

# [0] Data generating mode parameters
path_steps = 16
epochs = 20
path_type = "ik"
save_images = True
dataset_path = "../datasets/"
dir_name = "3dof-arm-grid/"
save_path = dataset_path + dir_name + "images/"
textfile_name = "joint_state.txt"
textfile_path = dataset_path + dir_name + textfile_name
error_text = dataset_path + dir_name + "error.txt"
image_counter = 1
pos_type = "grid"
grid_range = 7

# [1] Control mode parameters
num_episodes = 100
steps_per_episode = 50
load_model_path = "../log/3dof_regressor_combined/model.ckpt"

# Shared parameters
num_joints = 3
image_shape = (128, 128, 3)

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


def reset_joint_pos(joint_handles):
    joint1_pos = np.random.uniform(0.25 * -np.pi, 0.25 * np.pi)
    joint2_pos = np.random.uniform(0.5 * -np.pi, 0)
    joint3_pos = np.random.uniform(0.5 * -np.pi, 0)
    vrep.simxSetJointPosition(clientID, joint_handles[0], joint1_pos, vrep.simx_opmode_oneshot)
    vrep.simxSetJointPosition(clientID, joint_handles[1], joint2_pos, vrep.simx_opmode_oneshot)
    vrep.simxSetJointPosition(clientID, joint_handles[2], joint3_pos, vrep.simx_opmode_oneshot)
    config = [joint1_pos, joint2_pos, joint3_pos]
    return config

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

if mode == 0:
    # Open textfile
    textfile = open(textfile_path, "w")
    errorfile = open(error_text, "w")
    errorfile.write("cx cy cz j1 j2 j3\n")

    err_count = 0
    success_count = 0

    if pos_type == "random":
        while success_count < epochs:
            print("Epoch: {}/{}".format(success_count + 1, epochs))
            # Reset cube pos
            cube_x_pos = np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 0.3)
            cube_y_pos = np.random.uniform(0.1, 0.5)
            cube_z_pos = init_cube_pos[2]
            vrep.simxSetObjectPosition(clientID, cube, -1, [cube_x_pos, cube_y_pos, cube_z_pos], vrep.simx_opmode_oneshot)

            # Reset joint pos
            # jh[0]/joint1, range: [-1/2*pi, 1/2*pi]
            # jh[1]/joint2, range: [-3/4*pi, 0]
            # jh[2]/joint3, range: [-3/4*pi, 0]
            current_config = reset_joint_pos(joint_handles)
            if path_type == 'ik':
                err, ikpath = generate_ikpath()
                if err != 0:
                    err_count += 1
                    print("Non successful pathfinding, redoing epoch...")
                else:
                    success_count += 1
                follow_path(ikpath, joint_handles, vs_diag, distance_handle, textfile)
            elif path_type == 'interpolate':
                end_config = get_end_config()
                follow_interpolate_path(current_config, end_config, joint_handles)
            else:
                print "Not a valid path type"

    elif pos_type == "grid":
        cube_x_grid = np.linspace(-0.3, 0.3, grid_range)
        cube_y_grid = np.linspace(0.15, 0.45, grid_range)
        cz = init_cube_pos[2]

        joint1_grid = np.linspace(-0.5 * np.pi, 0.5 * np.pi, grid_range)
        joint2_grid = np.linspace(-0.75 * np.pi, 0, grid_range)
        joint3_grid = np.linspace(-0.75 * np.pi, 0, grid_range)

        epoch_counter = 1
        epochs = grid_range ** 5
        for cx in cube_x_grid:
            for cy in cube_y_grid:
                for j1 in joint1_grid:
                    for j2 in joint2_grid:
                        for j3 in joint3_grid:
                            print("Epoch {}/{}".format(epoch_counter, epochs))
                            epoch_counter += 1
                            vrep.simxSetObjectPosition(clientID, cube, -1, [cx, cy, cz],
                                                       vrep.simx_opmode_oneshot)
                            vrep.simxSetJointPosition(clientID, joint_handles[0], j1,
                                                      vrep.simx_opmode_oneshot)
                            vrep.simxSetJointPosition(clientID, joint_handles[1], j2,
                                                      vrep.simx_opmode_oneshot)
                            vrep.simxSetJointPosition(clientID, joint_handles[2], j3,
                                                      vrep.simx_opmode_oneshot)
                            err, ikpath = generate_ikpath()
                            if err != 0:
                                err_count += 1
                                errorfile.write(str(cx) + " " + str(cy) + " " + str(cz) + " " + str(j1) + " " + str(j2) + " " + str(j3) + "\n")
                            else:
                                follow_path(ikpath, joint_handles, vs_diag, distance_handle, textfile)
    else:
        print "Not a valid generator type"

    # Close textfile
    textfile.close()
    errorfile.close()

    print("Error count:", err_count)

elif mode == 1:
    print("Control mode")
    import tensorflow as tf
    sess = tf.Session()
    saver = tf.train.import_meta_graph(load_model_path + ".meta")
    saver.restore(sess, load_model_path)
    image_op = tf.get_collection("image")[0]
    pred_op = tf.get_collection("pred")[0]

    min_distance_list = []
    final_distance_list = []
    for episode in range(num_episodes):
        print("Episode {}".format(episode + 1))
        config = reset_joint_pos(joint_handles)

        # Reset cube pos
        cube_x_pos = np.random.choice([-1.0, 1.0]) * np.random.uniform(0., 0.3)
        cube_y_pos = np.random.uniform(0.15, 0.45)
        cube_z_pos = init_cube_pos[2]
        vrep.simxSetObjectPosition(clientID, cube, -1, [cube_x_pos, cube_y_pos, cube_z_pos], vrep.simx_opmode_oneshot)

        err, distance = vrep.simxReadDistance(clientID, distance_handle, vrep.simx_opmode_blocking)
        min_distance = distance

        for step in range(steps_per_episode):
            err, resolution, image = vrep.simxGetVisionSensorImage(clientID, vs_diag, 0, vrep.simx_opmode_blocking)
            image = np.flip(np.array(image, dtype=np.uint8).reshape((1,) + image_shape), axis=1)
            image = (image / 127.5) - 1
            pred = sess.run(pred_op, feed_dict={image_op: image})
            pred = pred[0]
            display(image[0])

            for j in range(num_joints):
                err, jp = vrep.simxGetJointPosition(clientID, joint_handles[j], vrep.simx_opmode_blocking)
                err = vrep.simxSetJointPosition(clientID, joint_handles[j], jp + pred[j], vrep.simx_opmode_oneshot)

            err, distance = vrep.simxReadDistance(clientID, distance_handle, vrep.simx_opmode_blocking)
            if distance < min_distance:
                min_distance = distance

            if (step + 1) == steps_per_episode:
                final_distance_list.append(distance)

            vrep.simxSynchronousTrigger(clientID)
            vrep.simxGetPingTime(clientID)

        min_distance_list.append(min_distance)

    min_dist_threshold = 0.1
    print "Success rate: ", 1.0 * np.sum(np.array(min_distance_list) <= min_dist_threshold) / len(min_distance_list)
    print "Total episodes: ", len(min_distance_list)
    print "Average min distance: %.3f" % np.mean(min_distance_list)
    print "Standard deviation: %.3f" % np.std(min_distance_list)
    print "Average final distance: %.3f" % np.mean(final_distance_list)
    print "Standard deviation: %.3f" % np.std(final_distance_list)

err = vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)

