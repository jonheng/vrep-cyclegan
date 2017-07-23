import vrep
import tensorflow as tf
from vrep_utils import VrepConnection
from image_utils import *


def control(load_model_path, total_episodes=10, im_function=False):
    # Initialize connection
    connection = VrepConnection()
    connection.synchronous_mode()
    connection.start()

    # Load tf model
    sess = tf.Session()
    saver = tf.train.import_meta_graph(load_model_path + ".meta")
    saver.restore(sess, load_model_path)
    # pred_op = tf.get_collection("pred")[0]
    # input_op = tf.get_collection("image_batch")[0]
    pred_op = tf.get_collection("reg_pred")[0]
    input_op = tf.get_collection("reg_input")[0]

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
    err, distanceToCubeHandle = vrep.simxGetDistanceHandle(clientID, "tipToCube", vrep.simx_opmode_blocking)
    err, distanceToCube = vrep.simxReadDistance(clientID, distanceToCubeHandle, vrep.simx_opmode_streaming)

    err, distanceToTargetHandle = vrep.simxGetDistanceHandle(clientID, "tipToTarget", vrep.simx_opmode_blocking)
    err, distanceToTarget = vrep.simxReadDistance(clientID, distanceToTargetHandle, vrep.simx_opmode_streaming)

    # numpy print options
    np.set_printoptions(precision=5, suppress=True)

    # Step while IK movement has not begun
    returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)
    while (signalValue == 0):
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)
        returnCode, signalValue = vrep.simxGetIntegerSignal(clientID, "ikstart", vrep.simx_opmode_streaming)


    # Iterate over total steps desired
    current_episode = 0
    step_counter = 0
    while current_episode < total_episodes + 1:
        # obtain current episode
        inputInts = []
        inputFloats = []
        inputStrings = []
        inputBuffer = bytearray()
        episode_table = []
        while len(episode_table) == 0:
            err, episode_table, _, _, _ = vrep.simxCallScriptFunction(clientID, 'Mico',
                                                                      vrep.sim_scripttype_childscript,
                                                                      'episodeCount', inputInts,
                                                                      inputFloats, inputStrings,
                                                                      inputBuffer, vrep.simx_opmode_blocking)
        if episode_table[0] > current_episode:
            step_counter = 0
            print "Episode: ", episode_table[0]
        current_episode = episode_table[0]
        step_counter += 1

        # 1. Obtain image from vision sensor
        err, resolution, img = vrep.simxGetVisionSensorImage(clientID, v1, 0, vrep.simx_opmode_buffer)
        image_shape = (resolution[0], resolution[1], 3)
        img = np.array(img, dtype=np.uint8)
        img = np.resize(img, (1,) + image_shape)  # resize into proper shape for input to neural network
        img = img.astype('float32')
        img = np.flip(img, axis=1)
        img = (img - 127.0) / 128.0  # convert image into a range of [-1, 1]

        original_img = img
        if im_function:
            #img = adjust_gamma(img, gamma=2)
            img = tint_images(img, [0.25, 0.5, 0.75])
        if (current_episode==1) and (step_counter==1):
            print "Displaying before and after image transformation"
            display_2images(original_img[0], img[0])
            plt.show()


        # 2. Pass into neural network to get joint velocities
        jointvel = sess.run(pred_op, feed_dict={input_op: img})
        jointvel = jointvel[0]
        stepsize = 1
        jointvel *= stepsize

        # 3. Apply joint velocities to arm in V-REP
        for j in range(6):
            err, jp = vrep.simxGetJointPosition(clientID, jhList[j], vrep.simx_opmode_buffer)
            jointpos[j] = jp
            err = vrep.simxSetJointPosition(clientID, jhList[j], jointpos[j] + jointvel[j], vrep.simx_opmode_oneshot)

        # Obtain distance
        err, distanceToCube = vrep.simxReadDistance(clientID, distanceToCubeHandle, vrep.simx_opmode_buffer)
        err, distanceToTarget = vrep.simxReadDistance(clientID, distanceToTargetHandle, vrep.simx_opmode_buffer)

        # Print statements
        # print "Step: ", step_counter
        print "Joint velocities: ", jointvel, " Abs sum: %.3f" % np.sum(np.absolute(jointvel))
        # print "Distance to cube: %.3f" % distanceToCube, ", Distance to target: %.3f" % distanceToTarget

        # trigger next step and wait for communication time
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)

    # obtain performance metrics
    inputInts = []
    inputFloats = []
    inputStrings = []
    inputBuffer = bytearray()
    err, minDistStep, minDist, _, _ = vrep.simxCallScriptFunction(clientID, 'Mico',
                                                                  vrep.sim_scripttype_childscript,
                                                                  'performanceMetrics', inputInts,
                                                                  inputFloats, inputStrings,
                                                                  inputBuffer, vrep.simx_opmode_blocking)

    if res == vrep.simx_return_ok:
        print "Min distance: ", minDist
        minThreshold = 0.2
        print "Success rate: ", 1.0*np.sum(np.array(minDist) <= minThreshold)/len(minDist)
        print "Total episodes: ", len(minDist)
        print "Average min distance: %.3f" % np.mean(minDist)
        print "Standard deviation: %.3f" % np.std(minDist)
    # other performance metrics such as success % can be defined (i.e. % reaching certain min threshold)

    # stop the simulation:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)

    # close tf session
    sess.close()

    return

if __name__=="__main__":
    control("../log/a2b2a_rgen/model.ckpt", total_episodes=10, im_function=True)
