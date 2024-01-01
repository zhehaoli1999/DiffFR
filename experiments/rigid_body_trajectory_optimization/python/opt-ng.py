import argparse
from argparse import RawTextHelpFormatter
import pysplishsplash as sph
import numpy as np
from numpy.linalg import norm
from utils import * 
import os
import nevergrad as ng

os.environ['KMP_DUPLICATE_LIB_OK']='True'

supported_optimizers = {'CMAES': ng.optimizers.ParametrizedCMA(scale=1.0, popsize=10),
                        '1+1ES': ng.optimizers.ParametrizedOnePlusOne()}

def get_parser():
    parser = argparse.ArgumentParser(description='SPlisHSPlasH', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--scene', type=str, help='scene file')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of boundary samples/maps.')
    parser.add_argument('--no-initial-pause', action='store_true', help=
                        'Disable initial pause when starting the simulation.')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI.')
    parser.add_argument('--load-fluid-pos', action='store_true', help='only load fluid pos from state file')
    parser.add_argument('--load-fluid-pos-and-vel', action='store_true',
                        help='load fluid pos and vel from state file')
    parser.add_argument('--stopAt', type=float, default=-1.0, help='Sets or overwrites the stopAt parameter of the scene.')
    parser.add_argument('--state-file', type=str, default='', help='State file (state_<time>.bin) that should be loaded.')
    parser.add_argument('--output-dir', type=str, default='', help='Output directory for log file and partio files.')
    parser.add_argument('--param', type=str, default='', help='Sets or overwrites a parameter of the scene.\n\n' 
					  '- Setting a fluid parameter:\n\t<fluid-id>:<parameter-name>:<value>\n'
					  '- Example: --param Fluid:viscosity:0.01\n\n'
					  '- Setting a configuration parameter:\n\t<parameter-name>:<value>\n'
					  '- Example: --param cflMethod:1\n')
    parser.add_argument('--positionWeight', type=float, default=1.0)
    parser.add_argument('--rotationWeight', type=float, default=1.0)
    parser.add_argument('--penaltyWeight', type=float, default=1.0)
    parser.add_argument('--maxIter', type=int, default=100)
    parser.add_argument('--taskType', type=str, default='bottle-flip', help='Choose from: \'bottle-flip\', \'stone-skipping\' or \'high-diving\'')
    parser.add_argument('--optimizer', type=str, default='CMAES', help=f'Supported: {list(supported_optimizers.keys())}')
    parser.add_argument('--load-optimizer-file', type=str, default='')
    return parser

parser = get_parser()
args = parser.parse_args()

if args.taskType not in ['bottle-flip', 'stone-skipping', 'high-diving', 'water-rafting']:
    print(f"{red_head} Unsupported task type!{color_tail}")
    quit()

opt_options = {'bottle-flip': (True, True), 'stone-skipping': (True, True), 'high-diving': (False, True), 'water-rafting': (True, True)}

is_v_opt, is_omega_opt = opt_options[args.taskType]
has_penalty = args.taskType == 'stone-skipping'
has_position_loss = args.taskType in ['bottle-flip', 'stone-skipping', 'water-rafting']
has_rotation_loss = args.taskType in ['bottle-flip', 'high-diving', 'water-rafting']

assert(args.optimizer in supported_optimizers.keys())

base = sph.Exec.SimulatorBase()
base.init(sceneFile=args.scene, useGui=not args.no_gui, initialPause=not args.no_initial_pause,\
          useCache=not args.no_cache, stopAt=args.stopAt, stateFile=args.state_file, \
          loadFluidPos = args.load_fluid_pos, loadFluidPosAndVel=args.load_fluid_pos_and_vel, \
          outputDir=args.output_dir, param=args.param)
if not args.no_gui:
    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)
base.initSimulation()
sim = sph.Simulation.getCurrent()
timestep = sim.getTimeStep()

parameter_v = ng.p.Array(shape=(3,))
parameter_v.value = timestep.get_init_v_rb(1)
parameter_omega = ng.p.Array(shape=(3,))
parameter_omega.value = timestep.get_init_omega_rb(1)
if is_v_opt and is_omega_opt:
    parameters = ng.p.Instrumentation(init_v=parameter_v, init_omega=parameter_omega)
elif is_v_opt:
    parameters = ng.p.Instrumentation(init_v=parameter_v)
else:
    parameters = ng.p.Instrumentation(init_omega=parameter_omega)
parameters.random_state.seed(12)
optimizer = supported_optimizers[args.optimizer](parametrization=parameters, budget=args.maxIter)
if args.load_optimizer_file != '':
    optimizer.load(args.load_optimizer_file)
optimizer.enable_pickling()

timestep.add_log(f"nevergrad parameter value: {parameters.value}")

timestep.add_log(f"{green_head}{args}{color_tail}")

def log_init_info():
    info = ''
    if is_v_opt:
        global init_v
        info += f"init_v: {init_v}\n"

    if is_omega_opt:
        global init_omega
        info += f"init_omega: {init_omega}\n"
    
    return info

def set_init_info(guess):
    if is_v_opt:
        global init_v
        init_v = guess.kwargs["init_v"]
        if args.taskType in ['water-rafting']:
            init_v[1] = -1.
        timestep.set_init_v_rb(1, init_v)
    if is_omega_opt:
        global init_omega
        init_omega = guess.kwargs["init_omega"]
        timestep.set_init_omega_rb(1, init_omega)

guess = optimizer.ask()
guess_number = 1
init_v = 0
init_omega = 0
set_init_info(guess)

timestep.add_log(f"{green_head}---------- guess [{guess_number}] ---------\n{log_init_info()}{color_tail}")

rotationWeight = args.rotationWeight
positionWeight = args.positionWeight
penaltyWeight = args.penaltyWeight

def position_loss():
    loss_x = 0
    if has_position_loss:
        bm = timestep.get_boundary_model(1)
        final_x_rb = bm.get_position_rb()
        target_x = timestep.get_target_x(1)
        loss_x = 0.5 * positionWeight * norm(target_x - final_x_rb)**2
        timestep.set_loss_x(loss_x)
        timestep.add_log(f"{green_head} loss_x = {loss_x}{color_tail}")
    return loss_x

def penalty_loss():
    loss_penalty = 0
    if has_penalty:
        upperbound = -5.
        loss_penalty = penaltyWeight * max(timestep.get_init_v_rb(1)[1] - upperbound, 0)
        timestep.add_log(f"{green_head} loss_penalty = {loss_penalty}{color_tail}")
    return loss_penalty

def rotation_loss():
    loss_rotation = 0
    if has_rotation_loss:
        bm = timestep.get_boundary_model(1)
        final_quaternion_rb= bm.get_quaternion_rb_vec4()
        target_quaternion =timestep.get_target_quaternion_vec4(1) # numpy array
        if args.taskType == "bottle-flip": 
            loss_rotation = 0.5 * rotationWeight * norm(final_quaternion_rb - target_quaternion)**2
        elif args.taskType in ["high-diving", "water-rafting"]: 
            delta_rotation1 = final_quaternion_rb - target_quaternion 
            delta_rotation2 = final_quaternion_rb + target_quaternion 
            loss_rotation1 = 0.5 * rotationWeight * norm(delta_rotation1)**2
            loss_rotation2 = 0.5 * rotationWeight * norm(delta_rotation2)**2
            if loss_rotation1 < loss_rotation2:
                loss_rotation = loss_rotation1
            else:
                loss_rotation = loss_rotation2
        timestep.set_loss_rotation(loss_rotation)
        timestep.add_log(f"{green_head} loss_rotation = {loss_rotation}{color_tail}")
    return loss_rotation

def get_loss():
    loss = position_loss() + rotation_loss() + penalty_loss()
    timestep.set_loss(loss)
    timestep.add_log(f"{green_head} loss = {loss}{color_tail}")
    return loss

get_final_answer = False

def time_step_callback():
    global guess, guess_number, get_final_answer
    if timestep.is_trajectory_finish_callback():
        loss = get_loss()
        sph.Utilities.Timing.printAverageTimes()
        sph.Utilities.Timing.printTimeSums()
        if get_final_answer:
            optimizer.dump(f'{args.output_dir}/ng_state_{guess_number}.pkl')
            timestep.add_log(f"{green_head}---------- final answer ---------\nloss: {loss}\n{color_tail}")
            if args.no_gui:
                base.setValueFloat(base.STOP_AT, 1e-5)
                base.reset()
            else:
                gui.stop()
            return
        timestep.add_log(f"{green_head}---------- guess [{guess_number}] ---------\nloss: {loss}\n{color_tail}")
        optimizer.tell(guess, loss)

        if guess_number < args.maxIter:
            guess_number += 1
            if loss < 1e-7:
                get_final_answer = True
                guess = optimizer.recommend()
            else:
                guess = optimizer.ask()
            set_init_info(guess)
            timestep.add_log(f"{green_head}---------- guess [{guess_number}] ---------\n{log_init_info()}{color_tail}")
            base.reset()
        else:
            final_guess = optimizer.recommend()
            set_init_info(final_guess)
            timestep.add_log(f"{green_head}---------- final answer ---------\n{log_init_info()}{color_tail}")
            # base.setValueBool(base.PAUSE, True)
            base.reset()
            get_final_answer = True

base.setTimeStepCB(time_step_callback)
base.runSimulation()
base.cleanup()


