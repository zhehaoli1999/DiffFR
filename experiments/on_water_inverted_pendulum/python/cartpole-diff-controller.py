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
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.manual_seed(12)
#-----------------------------------------------

LONG_TRAJECTORY_LENGTH = 0
SIGNAL_INTERVAL = 0 # control signal interval in long trajectory
n_epoch = 0
signal_count = 0
max_epochs = 0

baffle_period = 2.0
AA = 0. # baffle amplitude 
ww = 0  # baffle frequency.

# global variables for fixed-step-number operations
last_signal_step = 0
tb_writer = None 
device = "cpu"

json_index = {"body": 2, "rod": 1}
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
    parser.add_argument('--state', type=str, default='', help='State file (state_<time>.bin) that should be loaded.')
    parser.add_argument('--output-dir', type=str, default='', help='Output directory for log file and partio files.')
    parser.add_argument('--param', type=str, default='', help='Sets or overwrites a parameter of the scene.\n\n' 
					  '- Setting a fluid parameter:\n\t<fluid-id>:<parameter-name>:<value>\n'
					  '- Example: --param Fluid:viscosity:0.01\n\n'
					  '- Setting a configuration parameter:\n\t<parameter-name>:<value>\n'
					  '- Example: --param cflMethod:1\n')	
    parser.add_argument('--temp-state-dir', type=str, default='', help='Output directory for temporary state files generated in training')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--baffle-period', type=float, default=2)
    parser.add_argument('--baffle-distance', type=float, default=0.5)
#    parser.add_argument('--baffle-period', type=float, default=1.5)
 #   parser.add_argument('--baffle-distance', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--s', type=int, default=6, help='seconds to run')
    parser.add_argument('--signal-interval', type=int, default=50)
    parser.add_argument('--test', type=str, default="", help='load saved model path')
    parser.add_argument('--continue_train', type=str, default="", help='load saved model path')
    return parser

parser = get_parser()
args = parser.parse_args()
max_epochs = args.max_epochs
baffle_period = args.baffle_period
LONG_TRAJECTORY_LENGTH = args.s * 1000 + 1
SIGNAL_INTERVAL = args.signal_interval
ww = 2.0 * np.pi / args.baffle_period
AA = args.baffle_distance * ww / 2.0
simulator_base = sph.Exec.SimulatorBase()
simulator_base.init(sceneFile=os.path.abspath(args.scene), 
                    useGui=not args.no_gui, 
                    initialPause=not args.no_initial_pause,\
                    useCache=not args.no_cache, 
                    stopAt=args.stopAt, stateFile=os.path.abspath(args.state), \
                    loadFluidPos = args.load_fluid_pos,
                    outputDir=os.path.abspath(args.output_dir), 
                    param=args.param)

gui = sph.GUI.Simulator_GUI_imgui(simulator_base)
simulator_base.setGui(gui)
sim = sph.Simulation.getCurrent()

#-----------------------------------------------
# Baffle velocity control 
#-----------------------------------------------
def SinModeVelocity(t):
    return AA * np.sin(ww * t)

def getLBaffleVelocity(t):
    return SinModeVelocity(t)

def getRBaffleVelocity(t):
    if t < baffle_period / 2. :
        return 0.0
    else:
        return SinModeVelocity(t)

def set_baffle_speed(t):
    sim = sph.Simulation.getCurrent()
    Lbaffle = sim.getBoundaryModel(3).getRigidBodyObject()
    Rbaffle = sim.getBoundaryModel(4).getRigidBodyObject()
    lvel = getLBaffleVelocity(t)
    rvel = getRBaffleVelocity(t)
    if Lbaffle.isAnimated():
        Lbaffle.setVelocity([lvel, 0, 0])
        # Lbaffle.setVelocity([-rvel, 0, 0])
        Lbaffle.animate()
    if Rbaffle.isAnimated():
        Rbaffle.setVelocity([rvel, 0, 0])
        # Rbaffle.setVelocity([-lvel, 0, 0])
        Rbaffle.animate()

#------------------------------------
# TODO: NN controller here 
#------------------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Controller(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.dim = 2 
        # target x 
        # ------------------------------- TODO: add time to state ------------------------------------------------
        self.intput_dim = 12 + 1 # time + 2 parts x 6 (2 for x, 2 for v, 1 for omega, 1 for rotation) 
        # ----------------------------------------------------------------------------------
        # v and omega 
        self.output_dim = 2 # the velocity of body
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.intput_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(), 
                nn.Linear(16, self.output_dim),
                # nn.Softmax(dim=0)
            )
        self.vx_min = -0.5
        self.vx_max = 0.5

        self.vy_min = -0.5
        self.vy_max = 0.
        
        # self.linear_relu_stack.apply(init_weights)
        
    def forward(self, x):
        x = torch.Tensor(x)
        x = x.to(torch.float)
        # control_signal = 2 * (self.linear_relu_stack(x) - 0.5)
        control_signal = self.linear_relu_stack(x) 
        control_signal = torch.clamp(control_signal, 
                         min = torch.Tensor([self.vx_min, self.vy_min]), 
                         max = torch.Tensor([self.vx_max, self.vy_max]))

        return control_signal 

def get_loss_and_grad(simulator):
    rod = get_robot_rod() 
    final_q_rod = rod.get_quaternion_rb_vec4()
    target_q_rod = simulator.get_target_quaternion_vec4(json_index['rod'])
    loss_rotation = 0.5 * (norm(final_q_rod - target_q_rod)**2)
    grad_loss_rotation = final_q_rod - target_q_rod

    final_x_rod = rod.get_position_rb()
    target_x_rod = simulator.get_target_x(json_index['rod'])
    loss_x = 0.5 * (norm(final_x_rod - target_x_rod)**2)
    grad_loss_x = final_x_rod - target_x_rod

    print(f"{green_head}final_q_rod = {final_q_rod}{color_tail}")
    print(f"{green_head}target_q_rod = {target_q_rod}{color_tail}")
    print(f"{green_head}final_x_rod = {final_x_rod}{color_tail}")
    print(f"{green_head}target_x_rod = {target_x_rod}{color_tail}")
    # simulator.set_loss(loss_rotation)
    # simulator.set_loss(loss_rotation)

    return loss_rotation,  grad_loss_rotation, loss_x, grad_loss_x

class Simulator_layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, control_signal):
        loss_list = []
        grad_list = []
        sim = sph.Simulation.getCurrent()
        timestep = sim.getTimeStep()

        # -----------------------------------------------------------------------------------
        # First, add control signal to robot
        # simulator_base.forwardFixedSteps(2)
        
        add_robot_body_vel(control_signal)
        timestep.add_log(f"{yellow_head}[next interval] delta v = {control_signal} {color_tail}")
        timestep.add_log(f'{green_head} robot body v: {get_robot_body_v()} {color_tail}')

        simulator_base.forwardFixedSteps(SIGNAL_INTERVAL)

        # -----------------------------------------------------------------------------------
        # First, compute loss and grad
        loss_rotation, grad_loss_rotation, loss_x, grad_loss_x = get_loss_and_grad(timestep) 
        # w = 0.5
        # loss = loss_rotation + w * loss_x 
        loss = loss_rotation 
        timestep.add_log(f"{green_head} loss_rotation = {loss_rotation}{color_tail}")
        timestep.add_log(f"{green_head} loss_x = {loss_x}{color_tail}")
        # timestep.add_log(f"{green_head} loss = {loss}{color_tail}")
    
        rod = get_robot_rod()

        grad_rotation_to_init_v = rod.get_grad_quaternion_to_v0_another_bm(json_index['body']).transpose() @ grad_loss_rotation
        # Note: loss_x and grad_x_to_init_v are not used!
        # grad_x_to_init_v = rod.get_grad_x_to_v0_another_bm(2).transpose() @ grad_loss_x
        print(f"grad_rotation_to_init_v = {grad_rotation_to_init_v}")
        # print(f"grad_x_to_init_v = {grad_x_to_init_v}")
        # grad_l_to_init_v = grad_rotation_to_init_v + w * grad_x_to_init_v
        grad_l_to_init_v = grad_rotation_to_init_v 

        grad = np.array([grad_l_to_init_v[0], grad_l_to_init_v[1]])
        timestep.add_log(f"{cyan_head}grad v = {grad}\n")

        grad_list.append(grad)
        loss_list.append(loss)

        # reset gradient 
        timestep.reset_gradient() # reset gradient here to make current step the 0th step in gradient computation
        # -----------------------------------------------------------------------------------
       
        avg_loss = np.average(loss_list)
        ctx.save_for_backward(torch.Tensor(np.array([grad_list])))
        return torch.Tensor([avg_loss])
        
    @staticmethod
    def backward(ctx, grad_output) :
        grad, = ctx.saved_tensors
        return grad, None 

class Robot(nn.Module):
    def __init__(self)-> None:
        super().__init__()
        self.controller = Controller()
        self.simulator_layer = Simulator_layer.apply

    def forward(self, robot_state):
        control_signal = self.controller(robot_state)
        loss = self.simulator_layer(control_signal)
        return loss

    def get_control_signal(self, robot_state):
        return self.controller(robot_state).detach().numpy()

    def add_control_signal_to_robot(self):
        sim = sph.Simulation.getCurrent()
        timestep = sim.getTimeStep()

        control_signal = self.get_control_signal(get_robot_state())
        timestep.add_log(f'{yellow_head} delta v = {control_signal} {color_tail}')
        add_robot_body_vel(control_signal)

robot_model = Robot()
optimizer = torch.optim.Adam(robot_model.parameters(), lr=args.lr)
# epoch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# --------------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------
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

def add_robot_body_vel(delta_v):
    body = get_robot_body() 
    body.set_velocity_rb(np.array([delta_v[0], delta_v[1], 0]) + body.get_velocity_rb())

def normalize(a):
    a /= np.linalg.norm(a)
    return a

def get_robot_state():
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

def get_robot_body_v():
    body = get_robot_body()
    return [body.get_velocity_rb()]

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
        return True

    return False
#------------------------------------
# Callback after each time step
#------------------------------------
def train_callback_function():
    time_manager = sph.TimeManager.getCurrent()
    t = time_manager.getTime()
    set_baffle_speed(t)

test_last_signal_step = 0
def test_callback_function():
    time_manager = sph.TimeManager.getCurrent()
    t = time_manager.getTime()
    set_baffle_speed(t)

    sim = sph.Simulation.getCurrent()
    timestep = sim.getTimeStep()

    current_step = timestep.get_step_count()
    if current_step == 2:
        robot_model.add_control_signal_to_robot()

    global test_last_signal_step
    if current_step - test_last_signal_step >= SIGNAL_INTERVAL:
        global signal_count, avg_loss 
        signal_count += 1
        robot_model.add_control_signal_to_robot() 

        # tb_writer.add_scalar('Loss/train', loss, signal_count) 
        timestep.add_log(f'signal at time step {current_step}')
        timestep.add_log(f'{cyan_head} robot v: {get_robot_body_v()} {color_tail}')
        test_last_signal_step = current_step # update last_signal_step

    # print(f"{red_head} {if_need_early_stop()} {color_tail}")

# --------------------------------------------------------------------------------------------
# training function 
# --------------------------------------------------------------------------------------------


def train_loop():
    global robot_model 
    sim = sph.Simulation.getCurrent()
    timestep = sim.getTimeStep()

    avg_loss = 0 
    signal_count = 0
    robot_model.train()
    current_step = timestep.get_step_count()
    simulator_base.forwardFixedSteps(2)
    while current_step < LONG_TRAJECTORY_LENGTH:
        robot_state = get_robot_state()
        loss = robot_model(robot_state) # compute loss and give new control signal 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss 
        signal_count += 1

        # tb_writer.add_scalar('Loss/train', loss, signal_count) 
        timestep.add_log(f'{green_head} signal No.{signal_count} at time step {current_step} {color_tail}')
        current_step = timestep.get_step_count()

    global n_epoch, tb_writer
    # avg_loss /= signal_count
    # tb_writer.add_scalar('Loss/train', avg_loss, signal_count*n_epoch) 
    tb_writer.add_scalar('Loss/train', avg_loss, n_epoch) 
    simulator_base.reset() 

    n_epoch += 1
    if n_epoch % 10 == 0:
        now = strftime("%Y-%m-%d-%H:%M:%S", localtime())
        torch.save(robot_model.state_dict(), f"./saved_models/{now}-cartpole-epoch{n_epoch}.pth")

    # epoch_lr_scheduler.step()

    timestep.add_log(f"====== Enter new epoch No.{n_epoch} ======")
    if n_epoch > max_epochs:
        tb_writer.close()
        gui.stop()
        return    
     
# --------------------------------------------------------------------------------------------

def main():
    global tb_writer  
    # -------- training -------------
    if args.test == "" and args.continue_train == "":
        tb_writer = SummaryWriter()
        simulator_base.initSimulationWithDeferredInit()
        sim = sph.Simulation.getCurrent()
        timestep = sim.getTimeStep()
        timestep.add_log(f"{green_head}{args}{color_tail}")

        timestep.add_log(f"long trajectory length = {LONG_TRAJECTORY_LENGTH}")
        timestep.add_log(f"control signal interval = {SIGNAL_INTERVAL}")

        simulator_base.setTimeStepCB(train_callback_function)
        for n_epoch in range(max_epochs):
            timestep.add_log(f"{green_head}Epoch {n_epoch+1}\n-------------------------------{color_tail}")
            train_loop()

    # -------- testing -------------
    elif args.test != "":
        simulator_base.initSimulation()
        sim = sph.Simulation.getCurrent()
        timestep = sim.getTimeStep()
        timestep.add_log(f"{green_head} ================ testing ================ {color_tail}")
        if args.test != "init":
            robot_model.load_state_dict(torch.load(args.test))
        robot_model.eval()
        simulator_base.setTimeStepCB(test_callback_function)
        simulator_base.runSimulation()
    # -------- continue training -------------
    else: 
        tb_writer = SummaryWriter()
        simulator_base.initSimulationWithDeferredInit()
        timestep = sim.getTimeStep()
        timestep.add_log(f"{green_head}{args}{color_tail}")

        timestep.add_log(f"long trajectory length = {LONG_TRAJECTORY_LENGTH}")
        timestep.add_log(f"control signal interval = {SIGNAL_INTERVAL}")

        robot_model.load_state_dict(torch.load(args.continue_train))
        simulator_base.setTimeStepCB(train_callback_function)
        for n_epoch in range(max_epochs):
            timestep.add_log(f"{green_head}Epoch {n_epoch+1}\n-------------------------------{color_tail}")
            train_loop()

    simulator_base.cleanup()

main()
