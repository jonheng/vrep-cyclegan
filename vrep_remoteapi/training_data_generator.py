import vrep
from vrep_utils import VrepConnection
import h5py
import numpy as np
import math


def joint_difference(initial_pos, end_pos):
    joint_vel = np.empty(6)
    for i in range(6):
        difference = end_pos[i] - initial_pos[i]
        if difference > math.pi:
            joint_vel[i] = difference - 2 * math.pi
        elif difference < -math.pi:
            joint_vel[i] = difference + 2 * math.pi
        else:
            joint_vel[i] = difference
    return joint_vel


def generate(save_path, number_episodes, steps_per_episode,
             jointvel_factor=0.5):

    # Initialize connection
    connection = VrepConnection()
    connection.synchronous_mode()
    connection.start()

    # Use client id from connection
    clientID = connection.clientID

    # Get joint handles
    jhList = [-1, -1, -1, -1, -1, -1]
    for i in range(6):
        err, jh = vrep.simxGetObjectHandle(clientID, "Mico_joint" + str(i + 1), vrep.simx_opmode_blocking)
        jhList[i] = jh

    # Initialize joint position
    jointpos = np.zeros(6)
    for i in range(6):
        err, jp = vrep.simxGetJointPosition(clientID, jhList[i], vrep.simx_opmode_streaming)
        jointpos[i] = jp

    # Initialize vision sensor
    res, v1 = vrep.simxGetObjectHandle(clientID, "vs1", vrep.simx_opmode_oneshot_wait)
    err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_streaming)
    vrep.simxGetPingTime(clientID)
    err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)

    # Initialize distance handle
    err, distance_handle = vrep.simxGetDistanceHandle(clientID, "tipToTarget", vrep.simx_opmode_blocking)
    err, distance_to_target = vrep.simxReadDistance(clientID, distance_handle, vrep.simx_opmode_streaming)

    # Initialize data file
    f = h5py.File(save_path, "w")
    episode_count = 0
    total_datapoints = number_episodes * steps_per_episode
    size_of_image = resolution[0] * resolution[1] * 3
    dset1 = f.create_dataset("images", (total_datapoints, size_of_image), dtype="uint")
    dset2 = f.create_dataset("joint_pos", (total_datapoints, 6), dtype="float")
    dset3 = f.create_dataset("joint_vel", (total_datapoints, 6), dtype="float")
    dset4 = f.create_dataset("distance", (total_datapoints, 1), dtype="float")

    # Step while IK movement has not begun
    returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)
    while (signalValue == 0):
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)
        returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)

    for i in range(total_datapoints):
        # At start of each episode, check if path has been found in vrep
        if (i % steps_per_episode) == 0:
            returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)
            while (signalValue == 0):
                vrep.simxSynchronousTrigger(clientID)
                vrep.simxGetPingTime(clientID)
                returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)
            episode_count += 1
            print "Episode: ", episode_count

        # obtain image and place into array
        err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)
        img = np.array(image, dtype=np.uint8)
        dset1[i] = img

        # obtain joint pos and place into array
        jointpos = np.zeros(6)
        for j in range(6):
            err, jp = vrep.simxGetJointPosition(clientID, jhList[j], vrep.simx_opmode_buffer)
            jointpos[j] = jp
        dset2[i] = jointpos

        # obtain distance and place into array
        distance = np.zeros(1)
        err, distance_to_target = vrep.simxReadDistance(clientID, distance_handle, vrep.simx_opmode_buffer)
        distance[0] = distance_to_target
        dset4[i] = distance

        # trigger next step and wait for communication time
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)

    # stop the simulation:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)

    # calculate joint velocities excluding final image
    for k in range(number_episodes):
        for i in range(steps_per_episode - 1):
            #jointvel = dset2[k*steps_per_episode + i+1]-dset2[k*steps_per_episode + i]
            jointvel = joint_difference(dset2[k*steps_per_episode + i], dset2[k*steps_per_episode + i+1])
            abs_sum = np.sum(np.absolute(jointvel))
            if abs_sum == 0:
                dset3[k * steps_per_episode + i] = np.zeros(6)
            else: # abs sum norm
                dset3[k * steps_per_episode + i] = jointvel/abs_sum * jointvel_factor * dset4[k * steps_per_episode + i]

    # close datafile
    f.close()

    return

if __name__=="__main__":
    print "Generating new dataset"
    generate("../datasets/d1.hdf5", 800, 25)