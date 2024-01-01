import pysplishsplash as sph
import argparse
from argparse import RawTextHelpFormatter
import torch
import torch.nn as nn
import os
import numpy as np
import quaternion as qu
from numpy.linalg import norm
import pickle as pk
import sys

from torch.optim import optimizer
# sys.path.append("./")
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_parser():
    parser = argparse.ArgumentParser(
        description='SPlisHSPlasH', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--scene', type=str, help='scene file')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching of boundary samples/maps.')
    parser.add_argument('--no-initial-pause', action='store_true',
                        help='Disable initial pause when starting the simulation.')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI.')
    parser.add_argument('--load-fluid-pos', action='store_true',
                        help='only load fluid pos from state file')
    parser.add_argument('--load-fluid-pos-and-vel', action='store_true',
                        help='load fluid pos and vel from state file')
    parser.add_argument('--stopAt', type=float, default=-1.0,
                        help='Sets or overwrites the stopAt parameter of the scene.')
    parser.add_argument('--state', type=str, default='',
                        help='State file (state_<time>.bin) that should be loaded.')
    parser.add_argument('--output-dir', type=str, default='',
                        help='Output directory for log file and partio files.')
    parser.add_argument('--param', type=str, default='', help='Sets or overwrites a parameter of the scene.\n\n'
                      '- Setting a fluid parameter:\n\t<fluid-id>:<parameter-name>:<value>\n'
                      '- Example: --param Fluid:viscosity:0.01\n\n'
                      '- Setting a configuration parameter:\n\t<parameter-name>:<value>\n'
                      '- Example: --param cflMethod:1\n')
    parser.add_argument('--lr-v', type=float, default=1.0)
    parser.add_argument('--lr-omega', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--rotationWeight', type=float, default=1.)
    parser.add_argument('--positionWeight', type=float, default=1.)
    parser.add_argument('--penaltyWeight', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--maxIter', type=int, default=150)
    parser.add_argument('--num_per_checkpoint', type=int, default=20)
    parser.add_argument('--need_checkpoint', action='store_true')
    parser.add_argument('--taskType', type=str, default='bottle-flip', help='Choose from: \'bottle-flip\', \'stone-skipping\', \'high-diving\',\'water-rafting \' ')
    parser.add_argument('--fullDerivative', action='store_true')
    parser.add_argument('--normalizeGrad', action='store_true')
    parser.add_argument('--gradientMode', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Choose from \'sgd\', \'adam\' ')
    # parser.add_argument('--act_rb', type=int, default=1)
    # parser.add_argument('--passive_rb', type=int, default=1)
    return parser


parser = get_parser()
args = parser.parse_args()
print_green(f"{args}")

position_weight = args.positionWeight
rotation_weight = args.rotationWeight
penalty_weight = args.penaltyWeight

if args.taskType not in ['bottle-flip', 'stone-skipping', 'high-diving', 'water-rafting']:
    print(f"{red_head} Unsupported task type!{color_tail}")
    quit()

# ----------------------------------------------------------------------------------------------

has_penalty = args.taskType == 'stone-skipping'
has_position_loss = args.taskType in [
    'bottle-flip', 'stone-skipping', 'water-rafting']
has_rotation_loss = args.taskType in [
    'bottle-flip', 'high-diving', 'water-rafting']


def position_loss():
    loss_x = 0
    grad_loss_x = np.array([0, 0, 0])
    if has_position_loss:
        bm = timestep.get_boundary_model(1)
        final_x_rb = bm.get_position_rb()
        target_x = timestep.get_target_x(1)
        loss_x = 0.5 * position_weight * norm(final_x_rb - target_x)**2
        timestep.set_loss_x(loss_x)
        timestep.add_log(f"{green_head} loss_x = {loss_x}{color_tail}")
        grad_loss_x = position_weight * (final_x_rb - target_x)
    return loss_x, grad_loss_x


def penalty_loss():
    loss_penalty = 0
    grad_loss_penalty = np.array([0, 0, 0])
    if has_penalty:
        upperbound = -5.
        loss_penalty = penalty_weight * \
            max(timestep.get_init_v_rb(1)[1] - upperbound, 0)
        grad_loss_penalty = penalty_weight * \
            np.array([0, 1, 0]) if loss_penalty > 0 else np.array([0., 0, 0])
        timestep.add_log(
            f"{green_head} loss_penalty = {loss_penalty}{color_tail}")
    return loss_penalty, grad_loss_penalty


def rotation_loss():
    loss_rotation = 0
    grad_loss_rotation = np.array([0, 0, 0, 0])
    if has_rotation_loss:
        bm = timestep.get_boundary_model(1)
        final_quaternion_rb = bm.get_quaternion_rb_vec4()
        target_quaternion = timestep.get_target_quaternion_vec4(
            1)  # numpy array

        if args.taskType == "bottle-flip": 
            loss_rotation = 0.5 * rotation_weight * \
                norm(final_quaternion_rb - target_quaternion)**2
            grad_loss_rotation = rotation_weight * \
                (final_quaternion_rb - target_quaternion)
        elif args.taskType in ["high-diving", "water-rafting"]: 
            delta_rotation1 = final_quaternion_rb - target_quaternion 
            delta_rotation2 = final_quaternion_rb + target_quaternion 
            loss_rotation1 = 0.5 * rotation_weight * norm(delta_rotation1)**2
            loss_rotation2 = 0.5 * rotation_weight * norm(delta_rotation2)**2

            if loss_rotation1 < loss_rotation2:
                loss_rotation = loss_rotation1
                grad_loss_rotation = rotation_weight * delta_rotation1
            else:
                loss_rotation = loss_rotation2
                grad_loss_rotation = rotation_weight * delta_rotation2

        timestep.set_loss_rotation(loss_rotation)
        timestep.add_log(
            f"{green_head} loss_rotation = {loss_rotation}{color_tail}")

    return loss_rotation, grad_loss_rotation


def get_loss_and_grad(simulator):
    loss_x, grad_loss_x = position_loss()
    loss_penalty, grad_loss_penalty = penalty_loss()
    loss_rotation, grad_loss_rotation = rotation_loss()
    loss = loss_x + loss_rotation + loss_penalty

    simulator.set_loss(loss)
    timestep.add_log(f"{green_head} loss = {loss}{color_tail}")

    return loss_x, loss_rotation, loss_penalty, grad_loss_x, grad_loss_rotation, grad_loss_penalty


def normalize(a):
    if np.linalg.norm(a) < 1e-10:
        timestep.add_log(
            f"{yellow_head}  Warning: gradient too small {color_tail}")
    a /= np.linalg.norm(a)
    return a

# ----------------------------------------------------------------------------------------------


class Simulator_layer_v(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_param, simulator):
        loss_x, loss_rotation, loss_penalty, grad_loss_x, grad_loss_rotation, grad_loss_penalty = get_loss_and_grad(
            simulator)

        sim = sph.Simulation.getCurrent()
        if(sim.useRigidGradientManager()):
            rb_grad_manager = base.getRigidBodyGradientManager()
            grad_x_to_init_v =rb_grad_manager.get_grad_x_to_v0(1,1).transpose() @ grad_loss_x
            grad_rotation_to_init_v = rb_grad_manager.get_grad_q_to_v0(1,1).transpose() @ grad_loss_rotation
        else:
            bm = simulator.get_boundary_model(1)
            grad_x_to_init_v = bm.get_grad_x_to_v0().transpose() @ grad_loss_x
            grad_rotation_to_init_v = bm.get_grad_quaternion_to_v0().transpose() @ grad_loss_rotation
        

        timestep.add_log(f"{yellow_head}grad_x_to_init_v_rb = {grad_x_to_init_v}\n\
            grad_rotation_to_init_v_rb = {grad_rotation_to_init_v}{color_tail}\n")

        # final grad
        grad_init_v_rb = grad_x_to_init_v + grad_loss_penalty
        if args.fullDerivative:
            grad_init_v_rb += grad_rotation_to_init_v
        timestep.add_log(f"{yellow_head}grad_init_v_rb = {grad_init_v_rb}")
        if args.normalizeGrad:
            grad_init_v_rb = normalize(grad_init_v_rb)
            timestep.add_log(
                f"{yellow_head}normalized_grad_init_v_rb = {grad_init_v_rb}")

        ctx.save_for_backward(torch.Tensor(np.array(grad_init_v_rb)))

        return torch.Tensor([loss_x + loss_penalty])

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad, None


# ----------------------------------------------------------------------------------------------
class Simulator_layer_omega(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_param, simulator):
        loss_x, loss_rotation, loss_penalty, grad_loss_x, grad_loss_rotation, grad_loss_penalty = get_loss_and_grad(
            simulator)

        # grad to init omega
        sim = sph.Simulation.getCurrent()
        if(sim.useRigidGradientManager()):
            rb_grad_manager = base.getRigidBodyGradientManager()
            grad_x_to_init_omega = rb_grad_manager.get_grad_x_to_omega0(1,1).transpose() @ grad_loss_x
            grad_rotation_to_init_omega = rb_grad_manager.get_grad_q_to_omega0(1,1).transpose() @ grad_loss_rotation
        else:
            bm = simulator.get_boundary_model(1)
            grad_x_to_init_omega = bm.get_grad_x_to_omega0().transpose() @ grad_loss_x
            grad_rotation_to_init_omega = bm.get_grad_quaternion_to_omega0(
            ).transpose() @ grad_loss_rotation

        timestep.add_log(f"{yellow_head}grad_rotation_to_init_omega_rb = {grad_rotation_to_init_omega}\n\
            grad_x_to_init_omega_rb = {grad_x_to_init_omega}{color_tail}\n")

        # final grad
        grad_init_omega_rb = grad_rotation_to_init_omega
        if args.fullDerivative:
            grad_init_omega_rb += grad_x_to_init_omega

        timestep.add_log(
            f"{yellow_head}grad_init_omega_rb = {grad_init_omega_rb}")
        if args.normalizeGrad:
            grad_init_omega_rb = normalize(grad_init_omega_rb)
            timestep.add_log(
                f"{yellow_head}normalized_grad_init_omega_rb = {grad_init_omega_rb}")

        ctx.save_for_backward(torch.Tensor(np.array(grad_init_omega_rb)))

        return torch.Tensor([loss_rotation])

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad, None


# ----------------------------------------------------------------------------------------------
class Simulator_layer_shared(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_param, simulator):
        loss_x, loss_rotation, loss_penalty, grad_loss_x, grad_loss_rotation, grad_loss_penalty = get_loss_and_grad(
            simulator)

        # grad to init v
        bm = simulator.get_boundary_model(1)

        grad_x_to_init_v = bm.get_grad_x_to_v0().transpose() @ grad_loss_x
        grad_x_to_init_omega = bm.get_grad_x_to_omega0().transpose(
        ) @ grad_loss_x

        timestep.add_log(
            f"{yellow_head}grad_x_to_init_v_rb = {grad_x_to_init_v}{color_tail}\n"
        )
        timestep.add_log(
            f"{yellow_head}grad_x_to_init_omega_rb = {grad_x_to_init_omega}{color_tail}\n"
        )

        grad_init_v_rb = grad_x_to_init_v + grad_loss_penalty
        grad_init_omega_rb = grad_x_to_init_omega
        if args.normalizeGrad:
            grad_init_v_rb = normalize(grad_init_v_rb)
            grad_init_omega_rb = normalize(grad_init_omega_rb)
            timestep.add_log(
                f"{yellow_head}normalized_grad_init_v_rb = {grad_init_v_rb}")
            timestep.add_log(
                f"{yellow_head}normalized_grad_init_omega_rb = {grad_init_omega_rb}")

        ctx.save_for_backward(
            torch.Tensor([grad_init_v_rb, grad_init_omega_rb]))

        return torch.Tensor([loss_x])

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad, None


# ----------------------------------------------------------------------------------------------
class Controller(nn.Module):
    def __init__(self, simulator) -> None:
        super().__init__()

        self.simulator = simulator
        init_v_rb = self.simulator.get_init_v_rb(1)
        init_omega_rb = self.simulator.get_init_omega_rb(1)

        if args.taskType in ['bottle-flip', 'water-rafting']:
            self.init_param_v = nn.parameter.Parameter(
                torch.Tensor(init_v_rb))
            self.sim_v = Simulator_layer_v.apply

            self.init_param_omega = nn.parameter.Parameter(
                torch.Tensor(init_omega_rb))
            self.sim_omega = Simulator_layer_omega.apply

        elif args.taskType == 'stone-skipping':
            self.init_param_shared = nn.parameter.Parameter(
                torch.Tensor([init_v_rb, init_omega_rb]))
            self.sim_shared = Simulator_layer_shared.apply

        elif args.taskType == 'high-diving':
            self.init_param_omega = nn.parameter.Parameter(
                torch.Tensor(init_omega_rb))
            self.sim_omega = Simulator_layer_omega.apply

    def forward(self):
        if args.taskType in ['bottle-flip', 'water-rafting']:
            loss_v = self.sim_v(self.init_param_v, self.simulator)
            loss_omega = self.sim_omega(self.init_param_omega, self.simulator)
            return [loss_v, loss_omega]
        elif args.taskType == 'stone-skipping':
            loss = self.sim_shared(self.init_param_shared, self.simulator)
            return [loss]
        elif args.taskType == 'high-diving':
            loss_omega = self.sim_omega(self.init_param_omega, self.simulator)
            return [loss_omega]


# ----------------------------------------------------------------------------------------------
iteration = 0

base = sph.Exec.SimulatorBase()
base.init(sceneFile=os.path.abspath(args.scene), 
          useGui=not args.no_gui, 
          initialPause=not args.no_initial_pause,
          useCache=not args.no_cache, 
          stopAt=args.stopAt, 
          stateFile=os.path.abspath(args.state),
          loadFluidPos=args.load_fluid_pos, 
          loadFluidPosAndVel=args.load_fluid_pos_and_vel,
          outputDir=os.path.abspath(args.output_dir), 
          param=args.param)
gui = sph.GUI.Simulator_GUI_imgui(base)
base.setGui(gui)

base.initSimulation()
sim = sph.Simulation.getCurrent()
timestep = sim.getTimeStep()
timestep.add_log(f"{green_head}{args}{color_tail}")
sim.setGradientMode(args.gradientMode)
timestep.add_log(
    f"{green_head} gradient mode = {sim.getGradientMode()} (0 for complete, 1 for incomplete, 2 for rigid) {color_tail}")
net = Controller(simulator=timestep)
net.train()

checkpoint_dir = base.getOutputPath() + "/checkpoint/"
os.makedirs(checkpoint_dir, exist_ok=True)

# base.setValueBool(base.STATE_EXPORT, True)
# base.setValueFloat(base.STATE_EXPORT_FPS, 60)
# state_export_dir = checkpoint_dir + f"state-iter0/"
# os.makedirs(state_export_dir, exist_ok=True)
# base.setStateExportPath(state_export_dir)
# ----------------------------------------------------------------------------------------------

optimizers = []

if args.taskType in ['bottle-flip', 'water-rafting']:
    if args.optimizer == 'sgd':
        optimizers = [torch.optim.SGD([{"params": net.init_param_v}], lr=args.lr_v, momentum=args.momentum),
                      torch.optim.SGD([{"params": net.init_param_omega}], lr=args.lr_omega, momentum=args.momentum)]
    elif args.optimizer == 'adam':
        optimizers = [torch.optim.Adam([{"params": net.init_param_v}], lr=args.lr_v, betas=(args.momentum, 0.999)),
                      torch.optim.Adam([{"params": net.init_param_omega}], lr=args.lr_omega, betas=(args.momentum, 0.999))]
    else:
        print_red(" not supported optimizer!")

elif args.taskType == 'stone-skipping':
    if args.optimizer == 'sgd':
        optimizers = [torch.optim.SGD(
            [{"params": net.init_param_shared}], lr=args.lr_v, momentum=args.momentum)]
    elif args.optimizer == 'adam':
        optimizers = [torch.optim.Adam(
            [{"params": net.init_param_shared}], lr=args.lr_v, betas=(args.momentum, 0.999))]
    else:
        print_red(" not supported optimizer!")

elif args.taskType == 'high-diving':
    if args.optimizer == 'sgd':
        optimizers = [torch.optim.SGD(
            [{"params": net.init_param_omega}], lr=args.lr_omega, momentum=args.momentum)]
    elif args.optimizer == 'adam':
        optimizers = [torch.optim.Adam(
            [{"params": net.init_param_omega}], lr=args.lr_omega, betas=(args.momentum, 0.999))]
    else:
        print_red(" not supported optimizer!")


# ----------------------------------------------------------------------------------------------
schedulers = []
for i in range(len(optimizers)):
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[i],
                                                       mode='min',
                                                       factor=0.5,
                                                       patience=args.patience)  # patience should be small
                      )

# ----------------------------------------------------------------------------------------------


def has_nan(array):
    return np.isnan(np.sum(array))


def time_step_callback():
    global timestep, net, optimizers, loss_list, iteration

    if timestep.is_trajectory_finish_callback():
        # do the optimization
        iteration += 1
        net_loss = net()
        print_green(
            f"===== iter {iteration}, loss = {net_loss} ========")
        # TODO: add termination criteria

        for i in range(len(optimizers)):
            optimizers[i].zero_grad()
            net_loss[i].backward()
            optimizers[i].step()
            schedulers[i].step(net_loss[i])  # automatic adjust learning rate

        new_init_param_v = 0
        new_init_param_omega = 0
        if args.taskType == 'bottle-flip':
            new_init_param_v = net.init_param_v.detach().numpy()
            net.simulator.set_init_v_rb(1, new_init_param_v)
            new_init_param_omega = net.init_param_omega.detach().numpy()
            net.simulator.set_init_omega_rb(1, new_init_param_omega)

        elif args.taskType == 'stone-skipping':
            new_init_param_v = net.init_param_shared.detach().numpy()[0]
            new_init_param_omega = net.init_param_shared.detach().numpy()[1]
            net.simulator.set_init_v_rb(
                1, new_init_param_v)  # update init param
            net.simulator.set_init_omega_rb(1, new_init_param_omega)

        elif args.taskType == 'high-diving':
            new_init_param_omega = net.init_param_omega.detach().numpy()
            net.simulator.set_init_omega_rb(1, new_init_param_omega)

        elif args.taskType == 'water-rafting':
            new_init_param_v = net.init_param_v.detach().numpy()
            new_init_param_v[1] = -1.
            net.simulator.set_init_v_rb(1, new_init_param_v)
            new_init_param_omega = net.init_param_omega.detach().numpy()
            # --------------------------------------
            # For comparison 
            # new_init_param_omega[0] = 0.
            # new_init_param_omega[2] = 0.
            # --------------------------------------
            net.simulator.set_init_omega_rb(1, new_init_param_omega)

        if has_nan(new_init_param_v) or has_nan(new_init_param_omega):
            timestep.add_log(f"{red_head} Nan error!{color_tail}")
            quit()

        base.reset()
        timestep.clear_all_callbacks()
        # base.setValueInt(base.PAUSE, 1)

        for i in range(len(optimizers)):
            timestep.add_log(
                f"{cyan_head}optimizer[{i}] lr = {optimizers[i].param_groups[0]['lr']}{color_tail}\n")

        # ------------------------------------------------------------
        # final termination
        if iteration > args.maxIter:
            quit()
        is_lr_small = True
        for i in range(len(optimizers)):
            if optimizers[i].param_groups[0]['lr'] > 1e-6:
                is_lr_small = False
        if is_lr_small:
            quit()

        # ------------------------------------------------------------
        # checkpoint storation to start from intermidiate iteration
        if args.need_checkpoint:
            num_per_checkpoint = args.num_per_checkpoint
            if iteration % num_per_checkpoint == 0:
                base.setValueBool(base.STATE_EXPORT, True)
                base.setValueFloat(base.STATE_EXPORT_FPS, 60)
                # create a new folder & output state
                # save pytorch checkpoint
                checkpoint_filename= f"checkpoint-{iteration}.pt"
                state_export_dir = checkpoint_dir + f"state-iter{iteration}/"
                os.makedirs(state_export_dir, exist_ok=True)
                base.setStateExportPath(state_export_dir)
                optimizer_states = {}
                for i in range(len(optimizers)):
                    optimizer_states[f"optimizer{i}"] = optimizers[i].state_dict() 

                torch.save({
                            'iteration': iteration,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer_states,  
                            'loss': net_loss,
                            }, checkpoint_dir + checkpoint_filename)
            else:
                base.setValueBool(base.STATE_EXPORT, False)
        
        # -----------------------------------------

def main():
    base.setTimeStepCB(time_step_callback)
    base.runSimulation()
    base.cleanup()

if __name__ == "__main__":
    main()
