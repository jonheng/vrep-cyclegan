import numpy as np
import matplotlib.pyplot as plt


def joint_difference(initial_pos, end_pos, num_joints=3):
    joint_vel = np.empty(num_joints)
    for i in range(num_joints):
        difference = end_pos[i] - initial_pos[i]
        if difference > np.pi:
            joint_vel[i] = difference - 2 * np.pi
        elif difference < -np.pi:
            joint_vel[i] = difference + 2 * np.pi
        else:
            joint_vel[i] = difference
    return joint_vel


def convert_joint_state_to_vel(joint_state_path, joint_vel_path, steps_per_episode, jv_factor=0.5):
    joint_state_file = open(joint_state_path, "r")
    joint_vel_file = open(joint_vel_path, "w")

    js_data = joint_state_file.readlines()
    total_steps = len(js_data)

    for step in range(total_steps):
        if ((step + 1) % steps_per_episode) == 0:
            joint_vel_file.write(str(step + 1) + " 0 0 0\n")
            continue
        # remove trailing eol characters and split string by whitespace, then convert them into an numpy float array
        # step_data[0] -> index
        # step_data[1] -> joint1 state
        # step_data[2] -> joint2 state
        # step_data[3] -> joint3 state
        # step_data[4] -> distance
        step_data = np.array(js_data[step].rstrip().split(" "), dtype=np.float)
        next_step_data = np.array(js_data[step + 1].rstrip().split(" "), dtype=np.float)

        joint_config = step_data[1:4]
        next_joint_config = next_step_data[1:4]
        joint_vel = joint_difference(joint_config, next_joint_config)

        abs_sum = np.sum(np.absolute(joint_vel))
        joint_vel = joint_vel / abs_sum * jv_factor * step_data[4]

        joint_vel = np.round(joint_vel, decimals=5)
        joint_vel_str = str(step + 1) + " " + " ".join(joint_vel.astype(str)) + "\n"
        joint_vel_file.write(joint_vel_str)

    joint_state_file.close()
    joint_vel_file.close()
    return


def plot_loss(log_dir, save_path, text_path_list=["train_loss.txt", "test_loss.txt"]):
    plt_lines = []
    plt_names = []
    for path in text_path_list:
        textfile = open(log_dir + path, "r")
        data = textfile.readlines()
        x = []
        y = []
        for line in data:
            line_data = line.rstrip().split(" ")
            x.append(int(line_data[0]))
            y.append(float(line_data[1]))
        plt_line, = plt.plot(x, y)
        plt_lines.append(plt_line)
        plt_names.append(path.rstrip(".txt"))
    plt.legend(plt_lines, plt_names)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    return


if __name__ == "__main__":
    print("Dataset utils main")
    convert_joint_state_to_vel("datasets/3dof-arm-grid/joint_state.txt",
                               "datasets/3dof-arm-grid/joint_vel.txt",
                               steps_per_episode=16)

    # for i in range(9, 19):
    #     number_unique_images = 2**i
    #     plot_loss("log/3dof_regressor_" + str(number_unique_images) + "/", save_path="plots/3dof_regressor_" + str(number_unique_images) + ".png")
    #     plt.show()
