import vrep


class VrepConnection:
    def __init__(self):
        # Closes all opened connections
        vrep.simxFinish(-1)

        # Connects to V-REP
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

    def synchronous_mode(self):
        vrep.simxSynchronous(self.clientID, True)

    def start(self):
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)


