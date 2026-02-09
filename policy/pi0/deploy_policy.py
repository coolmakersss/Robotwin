import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    #input_state = observation["joint_action"]["vector"]
    input_state = np.concatenate([observation["endpose"]["left_endpose"],
                        observation["endpose"]["left_gripper"].reshape(1),
                        observation["endpose"]["right_endpose"],
                        observation["endpose"]["right_gripper"].reshape(1)])

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step)


def eval(TASK_ENV, model, observation):

    #if model.observation_window is None:
    instruction = TASK_ENV.get_instruction()
    model.call(func_name="set_language",obs=instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.call(func_name="update_observation_window", obs = (input_rgb_arr, input_state))

    # ======== Get Action ========

    #actions = model.call(func_name='get_action')[:model.pi0_step]
    actions = model.call(func_name='get_action')[:50]
    print(actions[0])


    for action in actions:
        TASK_ENV.take_action(action, action_type="ee")
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        #model.update_observation_window(input_rgb_arr, input_state)
        model.call(func_name="update_observation_window", obs = (input_rgb_arr, input_state))

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
