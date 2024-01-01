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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_parser():
    parser = argparse.ArgumentParser(description='SPlisHSPlasH', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--scene', type=str, help='scene file')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of boundary samples/maps.')
    parser.add_argument('--no-initial-pause', action='store_true', help=
                        'Disable initial pause when starting the simulation.')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI.')
    parser.add_argument('--load-fluid-pos', action='store_true', help='only load fluid pos from state file')
    parser.add_argument('--load-fluid-pos-and-vel', action='store_true', help='load fluid pos and vel from state file')
    parser.add_argument('--stopAt', type=float, default=-1.0, help='Sets or overwrites the stopAt parameter of the scene.')
    parser.add_argument('--state', type=str, default='', help='State file (state_<time>.bin) that should be loaded.')
    parser.add_argument('--output-dir', type=str, default='', help='Output directory for log file and partio files.')
    parser.add_argument('--param', type=str, default='', help='Sets or overwrites a parameter of the scene.\n\n' 
					  '- Setting a fluid parameter:\n\t<fluid-id>:<parameter-name>:<value>\n'
					  '- Example: --param Fluid:viscosity:0.01\n\n'
					  '- Setting a configuration parameter:\n\t<parameter-name>:<value>\n'
					  '- Example: --param cflMethod:1\n')	
    parser.add_argument('--lr-v', type=float, default=1.0)
    parser.add_argument('--lr-omega', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight', type=float, default=3.)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--maxIter', type=int, default=150)
    parser.add_argument('--act', type=int, default=1)
    parser.add_argument('--passive', type=int, default=2)
    parser.add_argument('--num_per_checkpoint', type=int, default=10)
    return parser

parser = get_parser()
args = parser.parse_args()

print_green(f"{args}")

act_rb_index = args.act 
passive_rb_index = args.passive 

def get_loss_and_grad(simulator):
    bm = simulator.get_boundary_model(passive_rb_index)
    final_x_rb = bm.get_position_rb()
    target_x = simulator.get_target_x(passive_rb_index)

    loss_x = 0.5 * norm(final_x_rb - target_x)**2 
    loss = loss_x 

    simulator.set_loss(loss)
    simulator.set_loss_x(loss_x)

    grad_loss_x = (final_x_rb - target_x)

    return loss_x, grad_loss_x

class Simulator_layer_v(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_param, simulator):
        loss_x, grad_loss_x = get_loss_and_grad(simulator) 

        loss = loss_x 
        timestep.add_log(f"{green_head} loss_x = {loss_x}{color_tail}")
        timestep.add_log(f"{green_head} loss = {loss}{color_tail}")

        # # grad to init v
        rb_grad_manager = base.getRigidBodyGradientManager()
        # rigid_contact_solver = base.getBoundarySimulator().getRigidContactSolver(); # Here we have problem 

        grad_x_to_init_v =  rb_grad_manager.get_grad_x_to_v0(passive_rb_index, act_rb_index).transpose() @ grad_loss_x 
        grad_x_to_init_omega =  rb_grad_manager.get_grad_x_to_omega0(passive_rb_index, act_rb_index).transpose() @ grad_loss_x 
        timestep.add_log(f"{red_head} grad_x_to_v0 = {rb_grad_manager.get_grad_x_to_v0(passive_rb_index, act_rb_index)}\n")
        timestep.add_log(f"{red_head} grad_x_to_omega0 = {rb_grad_manager.get_grad_x_to_omega0(passive_rb_index, act_rb_index)}\n")
        grad_init_v_rb = grad_x_to_init_v 
        grad_init_omega_rb = grad_x_to_init_omega

        timestep.add_log(f"{yellow_head}grad_x_to_init_v_rb = {grad_x_to_init_v}\n")
        timestep.add_log(f"{yellow_head}grad_x_to_init_omega_rb = {grad_x_to_init_omega}\n")

        ctx.save_for_backward(torch.Tensor([grad_init_v_rb, grad_init_omega_rb]))

        return torch.Tensor([ loss_x ])

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad, None


class Controller(nn.Module):
    def __init__(self, simulator) -> None:
        super().__init__()
        
        self.simulator = simulator
        init_v_rb = self.simulator.get_init_v_rb(act_rb_index)
        init_omega_rb = self.simulator.get_init_omega_rb(act_rb_index)
        self.init_param_v = nn.parameter.Parameter(
            torch.Tensor([init_v_rb, init_omega_rb])) 

        self.sim_v = Simulator_layer_v.apply

    def forward(self):
        loss_v = self.sim_v(self.init_param_v, self.simulator)
        return loss_v 

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
net = Controller(simulator=timestep) 
net.train()

checkpoint_dir = base.getOutputPath() + "/checkpoint/"
os.makedirs(checkpoint_dir, exist_ok=True)

base.setValueBool(base.STATE_EXPORT, True)
base.setValueFloat(base.STATE_EXPORT_FPS, 60)
state_export_dir = checkpoint_dir + f"state-iter0/"
os.makedirs(state_export_dir, exist_ok=True)
base.setStateExportPath(state_export_dir)

# Ref: https://pytorch.org/docs/stable/optim.html
# optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)
# optimizer = torch.optim.Adagrad(net.parameters(), lr=1.)
# optimizer = torch.optim.SGD(net.parameters() ,lr = args.lr)
# optimizers = [ torch.optim.SGD([{"params": net.init_param_v}],lr = args.lr_v,
                               # momentum=args.momentum),]
optimizers = [ torch.optim.Adam([{"params": net.init_param_v}],lr = args.lr_v, betas=(0.1,0.999))]

# optimizer = torch.optim.SGD([{'params': net.init_param_v, 'lr': args.lr_v}, 
                             # {'params': net.init_param_omega, 'lr': args.lr_omega}], lr=1.0)

# optimizer = torch.optim.LBFGS(net.parameters() ,lr = 1.)
# loss_fn = nn.MSELoss() # L2 norm # Ref: https://pytorch.org/docs/stable/nn.html#loss-functions
schedulers = []
for i in range(len(optimizers)):
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[i],
                                                       mode='min',
                                                       factor=0.5,
                                                       patience=args.patience) # patience should be small 
                      )

def time_step_callback():
    global timestep, net, optimizer, iteration

    if timestep.is_trajectory_finish_callback():
        # do the optimization 
        iteration += 1 

        net_loss_v = net() 
        net_loss = [net_loss_v]
        print_green(f"===== iter {iteration}, loss = {net_loss} ========")
        # TODO: add termination criteria 
        
        for i in range(len(optimizers)):
            optimizers[i].zero_grad()
            net_loss[i].backward()
            optimizers[i].step()
            schedulers[i].step(net_loss[i]) # automatic adjust learning rate

        new_init_param_v, new_init_param_omega = net.init_param_v.detach().numpy()
        net.simulator.set_init_v_rb(act_rb_index, new_init_param_v) # update init param
        net.simulator.set_init_omega_rb(act_rb_index, new_init_param_omega) # update init param
         
        base.reset()
        timestep.clear_all_callbacks()
        # base.setValueInt(base.PAUSE, 1)

        for i in range(len(optimizers)):
            timestep.add_log(f"{cyan_head}optimizer[{i}] lr = {optimizers[i].param_groups[0]['lr']}{color_tail}\n")
        
        if iteration > args.maxIter:
            quit()

        # for i in range(len(optimizers)):
        lr0 = optimizers[0].param_groups[0]['lr']
        if lr0 < 1e-5:
            quit()

        # ------------------------------------------------------------
        # checkpoint storation to start from intermidiate iteration
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

    # bm_p = timestep.get_boundary_model(passive_rb_index)
    # bm_a = timestep.get_boundary_model(act_rb_index)
    # print(f"v passive = {bm_p.get_velocity_rb()}")
    # print(f"v act = {bm_a.get_velocity_rb()}")
    # print(f"x passive = {bm_p.get_position_rb()}")


def main():
    base.setTimeStepCB(time_step_callback)
    base.runSimulation()
    base.cleanup()

if __name__ == "__main__":
    main()
