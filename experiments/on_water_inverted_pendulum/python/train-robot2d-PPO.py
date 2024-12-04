import pysplishsplash as sph
import torch 
from torch import nn, softmax
import argparse 
from argparse import RawTextHelpFormatter
import numpy as np
from numpy.linalg import norm
from utils import *
import os 
import random
from time import localtime, strftime
from torch.utils.tensorboard.writer import SummaryWriter

from PPO import PPO
os.environ['KMP_DUPLICATE_LIB_OK']='True'

writer = SummaryWriter('PPO')

#---------------------------------------------
json_index = {"body": 2, "rod": 1}
## parameters to change
LONG_TRAJECTORY_LENGTH = 1000
SIGNAL_INTERVAL = 50 # control signal interval in long trajectory
SAMPLE_STEP = 10 * LONG_TRAJECTORY_LENGTH

has_continuous_action_space = True
state_dim=12+1
action_dim=2
max_ep_len = 256                   # max timesteps in one episode
max_training_timesteps = int(2.56e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 20        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 8           # log avg reward in the interval (in num timesteps)
save_model_freq = int(6.4e4)          # save model frequency (in num timesteps) 4

action_std = 0.5                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.01         # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1               # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = 10    # action_std decay frequency (in num timesteps)
#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 50            # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)
LOAD = False
#-----------------------------------------------

epochs = 0
max_epochs = 0
state_dir = None
saved_files = []

simulator_base = None
timestep = None
sim = None
gui = None
time_manager = None
baffle_period = 2.0
AA = 0. # baffle amplitude 
ww = 0  # baffle frequency.

target=0.0

# global variables for fixed-step-number operations
last_signal_step = 0

#-----------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser(description='SPlisHSPlasH', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--scene', type=str, help='scene file')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of boundary samples/maps.')
    parser.add_argument('--no-initial-pause', action='store_true', help=
                        'Disable initial pause when starting the simulation.')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI.')
    parser.add_argument('--load-fluid-pos', action='store_true', help='only load fluid pos from state file')
    parser.add_argument('--stopAt', type=float, default=-1.0, help='Sets or overwrites the stopAt parameter of the scene.')
    parser.add_argument('--state-file', type=str, default='', help='State file (state_<time>.bin) that should be loaded.')
    parser.add_argument('--output-dir', type=str, default='', help='Output directory for log file and partio files.')
    parser.add_argument('--param', type=str, default='', help='Sets or overwrites a parameter of the scene.\n\n' 
					  '- Setting a fluid parameter:\n\t<fluid-id>:<parameter-name>:<value>\n'
					  '- Example: --param Fluid:viscosity:0.01\n\n'
					  '- Setting a configuration parameter:\n\t<parameter-name>:<value>\n'
					  '- Example: --param cflMethod:1\n')	
    parser.add_argument('--temp-state-dir', type=str, default='', help='Output directory for temporary state files generated in training')
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--baffle-period', type=float, default=2.0)
    parser.add_argument('--baffle-distance', type=float, default=0.8)
    return parser

#-----------------------------------------------
# Baffle velocity control 
#-----------------------------------------------
def SinModeVelocity(t):
    return AA * np.sin(ww * t)

def getLBaffleVelocity(t):
    return SinModeVelocity(t)

def getRBaffleVelocity(t):
    if t < baffle_period / 2. :
        return 0.0;
    else:
        return SinModeVelocity(t)

def set_baffle_speed(sim,t):

    Lbaffle = sim.getBoundaryModel(3).getRigidBodyObject()
    Rbaffle = sim.getBoundaryModel(4).getRigidBodyObject()
    lvel = getLBaffleVelocity(t)
    rvel = getRBaffleVelocity(t)
    if Lbaffle.isAnimated():
        Lbaffle.setVelocity([lvel*1.0, 0, 0])
        # Lbaffle.setVelocity([-rvel, 0, 0])
        Lbaffle.animate()
    if Rbaffle.isAnimated():
        Rbaffle.setVelocity([rvel*1.0, 0, 0])
        # Rbaffle.setVelocity([-lvel, 0, 0])
        Rbaffle.animate()

#---------------------------------------------------
# Helper functions for generating training data  
#---------------------------------------------------
def pause_simulation():
    simulator_base.setValueBool(simulator_base.PAUSE, True)

def continue_simulation():
    simulator_base.setValueBool(simulator_base.PAUSE, False)

def save_current_state():
    t = time_manager.getTime()
    current_state_dir = f"{state_dir}/{epochs}"
    state_file_name = f"state_{t:.4f}.bin"
    current_state_file = current_state_dir + '/' + state_file_name
    saved_files.append(state_file_name)
    simulator_base.saveState(current_state_file)

def load_state(state_file):
    simulator_base.loadStateWithRigidExisted(state_file)

#------------------------------------
# TODO: PPO controller here 
#------------------------------------
class Controller:
    def forward(self, state_file):
        # TODO: somehow return controlling signal from given state
        return None

controller = Controller()

def get_robot_body():
    sim = sph.Simulation.getCurrent()
    timestep = sim.getTimeStep()
    body = timestep.get_boundary_model(json_index['body'])
    return body
def get_robot_rod():
    sim = sph.Simulation.getCurrent()
    timestep = sim.getTimeStep()
    rod = timestep.get_boundary_model(json_index['rod'])
    return rod
def get_controller_signal():
    return

def update_controller():
    return

def add_robot_body_vel(delta_v):
    body = get_robot_body() 
    body.set_velocity_rb(np.array([delta_v[0], delta_v[1], 0]) + body.get_velocity_rb())

def get_robot_state():
    global sim
    sim = sph.Simulation.getCurrent()
    timestep = sim.getTimeStep()
    body = get_robot_body()
    rod = get_robot_rod() 

    robot = [body, rod]

    state = []
    for r in robot:
        x, q, v, omega = r.get_position_rb(), r.get_quaternion_rb_vec4(),\
                        r.get_velocity_rb(), r.get_angular_velocity_rb()    
        # print(f"{yellow_head}{x,q,v,omega}{color_tail}")
        x = np.array([x[0], x[1]])
        q = np.array([q[0]])
        v = np.array([v[0], v[1]])
        omega = np.array([omega[2]])
        state += [x, q, v, omega] # For every part of robot, this is a 6x1 vector 
        
    # state =  np.concatenate(state)
    # return state.reshape((1, len(state)))# a 18x1 vector 
    time_manager = sph.TimeManager.getCurrent()
    # ------------------------------- TODO: add time into state -----------------------------------------------
    t = time_manager.getTime()
    t=0
    state += [np.array([t])]
    # ----------------------------------------------------------------------------------
    # return normalize(np.concatenate(state))
    return np.concatenate(state)
def if_need_early_stop():

    body = get_robot_body()
    rod = get_robot_rod()

    x_body = body.get_position_rb()[0]
    y_body = body.get_position_rb()[1]
    if x_body > 4.5 or x_body < -4.5 or y_body > 4 or y_body < 0:
        return True

    sim = sph.Simulation.getCurrent()
    timestep = sim.getTimeStep()
    q_rod = rod.get_quaternion_rb_vec4()
    target_q_rod = timestep.get_target_quaternion_vec4(json_index['rod'])
    # print(f"q_rod = {q_rod}")
    # print(f"target_q_rod = {target_q_rod}")
    # print(f"{norm(q_rod - target_q_rod)**2}")
    if norm(q_rod - target_q_rod)**2 > 0.7:
        return 1

    return 0
def get_rewards(simulator):
    rod = get_robot_rod() 
    final_q_rod = rod.get_quaternion_rb_vec4()
    target_q_rod = simulator.get_target_quaternion_vec4(json_index['rod'])
    loss_rotation = 0.5 * (norm(final_q_rod - target_q_rod)**2)
    return -loss_rotation
#------------------------------------
# Callback after each time step
#------------------------------------
def callback_function():
    global last_signal_step 
    global tot_step,tot_reward
    global ppo_agent
    global sim
    global epochs
    current_step = timestep.get_step_count()
    tot_step+=1
    t = time_manager.getTime()
    set_baffle_speed(sim, t)

    # -----------------------------------------------------------
    # Add control signal to robot every SIGNAL_INTERVAL timesteps
    # -----------------------------------------------------------
    if current_step - last_signal_step >= SIGNAL_INTERVAL:
        state=get_robot_state()
        action=ppo_agent.select_action(state)
        reward = get_rewards(timestep)
        tot_reward+=reward
        done=0#if_need_early_stop()
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        # TODO: set controlling signals
        add_robot_body_vel(action)
        print(f'signal [{current_step}]')
        last_signal_step = current_step # update last_signal_step
        print(f"{yellow_head}{get_robot_state()}{color_tail}")

    # -------------------------------------------------------------------------------------------------------
    # If not in train mode, run a complete long trajectory and store intermidiate state as training data
    # If the long trajectory ends, enter train mode and load a saved state as the start of a short horizon
    # -------------------------------------------------------------------------------------------------------
    if current_step >= LONG_TRAJECTORY_LENGTH:
        writer.add_scalar('rewards', tot_reward, epochs)
        tot_reward=0
        simulator_base.reset() 
        last_signal_step = 0 
        

        #torch.save(ppo_agent.policy.actor, 'Pmodel/model_%d.pth' % epochs)
        torch.save(ppo_agent.policy.actor, 'Pmodel/model_1.pth' )
        torch.save(ppo_agent.policy.critic, 'Cmodel/model_1.pth' )
        epochs += 1
        print_green(f"====== Enter new epoch No.{epochs} ======")
        '''
        if epochs > max_epochs:
            gui.stop()
            return 
        '''
        if has_continuous_action_space and epochs % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
    if tot_step % SAMPLE_STEP == 0:
        ppo_agent.update()
# --------------------------------------------------------------------------------------------

def main():
    parser = get_parser()
    args = parser.parse_args()

    global max_epochs
    max_epochs = args.max_epochs
    
    global tot_step,tot_reward
    tot_step=0
    tot_reward=0
    
    global ppo_agent
    ppo_agent = PPO(LOAD,state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    
    
    
    # ------------------------------------------------------
    # Initialize simulator 
    # ------------------------------------------------------
    global simulator_base, timestep, sim, gui, time_manager, baffle_period, AA, ww
    simulator_base = sph.Exec.SimulatorBase()
    simulator_base.init(sceneFile=args.scene, useGui=not args.no_gui, initialPause=not args.no_initial_pause,\
              useCache=not args.no_cache, stopAt=args.stopAt, stateFile=args.state_file, \
              loadFluidPos = args.load_fluid_pos,
              outputDir=args.output_dir, param=args.param)


    gui = sph.GUI.Simulator_GUI_imgui(simulator_base)
    simulator_base.setGui(gui)
    sim = sph.Simulation.getCurrent()
    simulator_base.initSimulation()
    simulator_base.setTimeStepCB(callback_function)
    
    
    timestep = sim.getTimeStep()
    timestep.add_log(f"{green_head}{args}{color_tail}")
    time_manager = sph.TimeManager.getCurrent()

    baffle_period = args.baffle_period
    ww = 2.0 * np.pi / args.baffle_period
    AA = args.baffle_distance * ww / 2.0
    # ------------------------------------------------------

    global state_dir
    state_dir = args.temp_state_dir
    if state_dir == '':
        state_dir = simulator_base.getOutputPath() + "/tempState"
    print_green("Temporary state directory:\t", state_dir)
    # ------------------------------------------------------

    print_yellow(f"long trajectory length = {LONG_TRAJECTORY_LENGTH}")
    print_yellow(f"control signal interval = {SIGNAL_INTERVAL}")

    simulator_base.runSimulation()
    simulator_base.cleanup()

main()
