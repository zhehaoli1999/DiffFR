import pysplishsplash as sph 
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
import argparse 
from argparse import RawTextHelpFormatter
import numpy as np 
from numpy.linalg import norm
from utils import * 
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

timestep =None 

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

    parser.add_argument('--dim', type=int, default=3, help='dimension of simulator')
    parser.add_argument('--epoch', type=int, default=5, help='epoch of training')
    parser.add_argument('--batchsize', type=int, default=5, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight', type=float, default=3.0, help='weight of rotation loss')
    parser.add_argument('--num-train', type=int, default=1000, help='number of train data samples')
    parser.add_argument('--num-test', type=int, default=50, help='number of test data samples')
    parser.add_argument('--model-path', type=str, default='', help='test model with model file to load')
    parser.add_argument('--test_target', type=float, nargs='+', default=[3,3], help='test target positions')
    return parser

parser = get_parser()
args = parser.parse_args()
print_green(f"{args}")

class Actor(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__() 
        self.dim = dim 
        # target x 
        self.intput_dim = 3 if dim == 3 else 2  
        # v and omega 
        self.output_dim = (3 + 3) if dim == 3 else (2 + 1)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.intput_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(), 
                nn.Linear(32, self.output_dim)
            )
        self.v_min = -10 
        self.v_max = 10
        self.omega_min = -20
        self.omega_max = 20

    def forward(self, x):
        x = self.flatten(x)
        x = x.to(torch.float)
        control_signal = self.linear_relu_stack(x)
        if self.dim == 3:
            control_signal = torch.clamp(control_signal,
                             min = torch.Tensor([self.v_min, self.v_min, self.v_min,
                                                self.omega_min, self.omega_min, self.omega_min]), 
                             max = torch.Tensor([self.v_max, self.v_max, self.v_max,
                                                self.omega_max, self.omega_max, self.omega_max]))

        else:
            control_signal = torch.clamp(control_signal, 
                             min = torch.Tensor([self.v_min, self.v_min, self.omega_min]), 
                             max = torch.Tensor([self.v_max, self.v_max, self.omega_max]))

        return control_signal 

def normalize(a):
    a /= np.linalg.norm(a)
    return a
def get_loss_and_grad(simulator):
    bm = simulator.get_boundary_model(1)
    final_x_rb = bm.get_position_rb()
    final_quaternion_rb= bm.get_quaternion_rb_vec4()

    target_quaternion =simulator.get_target_quaternion_vec4(1) # numpy array
    target_x = simulator.get_target_x(1)

    loss_x = 0.5 * norm(target_x - final_x_rb)**2 
    w = args.weight
    loss_rotation = 0.5 * w * norm(final_quaternion_rb - target_quaternion)**2  

    loss = loss_x + loss_rotation

    simulator.set_loss(loss)
    simulator.set_loss_rotation(loss_rotation)
    simulator.set_loss_x(loss_x)

    grad_loss_x = 1. / norm(target_x)**2 * (final_x_rb - target_x)
    grad_loss_rotation = w * (final_quaternion_rb - target_quaternion)

    return loss_x, loss_rotation, grad_loss_x, grad_loss_rotation

class Simulator_layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, control_signal, target_pos, simulator, dim):
        batch_size = control_signal.shape[0]
        loss_list = []
        grad_list = []
        for b in range(batch_size):
            new_param = control_signal[b]
            target_x = target_pos[b]
            if dim == 3:
                init_v = new_param[:3]
                init_omega = new_param[3:6]
                target_x = np.array([target_x[0], target_x[1], target_x[2]])
            else:
                init_v = np.array([new_param[0], new_param[1], 0.])
                init_omega = np.array([0., 0., new_param[2]]) 
                target_x = np.array([target_x[0], target_x[1], 0])

            sim = sph.Simulation.getCurrent()
            timestep = sim.getTimeStep()
            timestep.set_init_v_rb(1, init_v)
            timestep.set_init_omega_rb(1, init_omega)
            timestep.set_target_x(1, target_x)
            timestep.add_log(f"{yellow_head}[next iter] target_x = {target_x} init_v = {init_v}, init_omega ={init_omega}{color_tail}")

            simulator.runNewTrajectory()

            # -----------------------------------------------------------------------------------
            loss_x, loss_rotation, grad_loss_x, grad_loss_rotation = get_loss_and_grad(timestep) 
            loss = loss_x + loss_rotation
            timestep.add_log(f"{green_head} loss_x = {loss_x}{color_tail}")
            timestep.add_log(f"{green_head} loss_rotation = {loss_rotation}{color_tail}")
            timestep.add_log(f"{green_head} loss = {loss}{color_tail}")
        
            bm = timestep.get_boundary_model(1)
            grad_x_to_init_v =  bm.get_grad_x_to_v0().transpose() @ grad_loss_x 
            grad_rotation_to_init_v = bm.get_grad_quaternion_to_v0().transpose() @ grad_loss_rotation
            timestep.add_log(f"{yellow_head}grad_x_to_init_v_rb = {grad_x_to_init_v}\n\
                grad_rotation_to_init_v_rb = {grad_rotation_to_init_v}{color_tail}\n")

            # final grad 
            # grad_init_v_rb = grad_x_to_init_v + grad_rotation_to_init_v
            grad_init_v_rb = grad_x_to_init_v 
            grad_init_v_rb = normalize(grad_init_v_rb)

            timestep.add_log(f"{yellow_head}grad_init_v_rb = {grad_init_v_rb}")

            # -----------------------------------------------------------------------------------

            grad_x_to_init_omega = bm.get_grad_x_to_omega0().transpose() @ grad_loss_x 
            grad_rotation_to_init_omega = bm.get_grad_quaternion_to_omega0().transpose() @ grad_loss_rotation

            # final grad 
            # grad_init_omega_rb = grad_x_to_init_omega + grad_rotation_to_init_omega
            grad_init_omega_rb = grad_rotation_to_init_omega
            grad_init_omega_rb = normalize(grad_init_omega_rb)

            timestep.add_log(f"{yellow_head}grad_rotation_to_init_omega_rb = {grad_rotation_to_init_omega}\n\
                grad_x_to_init_omega_rb = {grad_x_to_init_omega}{color_tail}\n")

            timestep.add_log(f"{yellow_head}grad_init_omega_rb = {grad_init_omega_rb}")

            # -----------------------------------------------------------------------------------

            if dim == 3:
                grad_list.append(np.array([grad_init_v_rb[0], grad_init_v_rb[1],grad_init_v_rb[2],
                                    grad_init_omega_rb[0], grad_init_omega_rb[1], grad_init_omega_rb[2]]))
            else:
                grad_list.append(np.array([grad_init_v_rb[0], grad_init_v_rb[1], grad_init_omega_rb[2]]))

            loss_list.append(loss)

        avg_loss = np.average(loss_list)
        ctx.save_for_backward(torch.Tensor(np.array([grad_list])))
        return torch.Tensor([avg_loss])
        
    @staticmethod
    def backward(ctx, grad_output) :
        grad, = ctx.saved_tensors
        return grad, None, None, None

class Robot(nn.Module):
    def __init__(self, dim, simulator) -> None:
        super().__init__()
        self.dim = dim
        self.actor = Actor(dim)
        self.simulator = simulator
        self.simulator_layer = Simulator_layer.apply

    def forward(self, target_pos):
        control_signal = self.actor(target_pos)
        loss = self.simulator_layer(control_signal, target_pos, self.simulator, self.dim)
        return loss

class targetPointDataset(Dataset):
    def __init__(self, target_positions) -> None:
        super().__init__()
        self.target_positions = target_positions

    def __len__(self):
        return len(self.target_positions)

    def __getitem__(self, idx):
        return self.target_positions[idx]


def generate_dataset(dim, num_train_samples, num_test_samples):
    start_x = 0.0
    start_y = -1.0
    end_x = 5.0
    end_y = 5.0
    point_x = np.linspace(start_x, end_x, num=50)
    point_y = np.linspace(start_y, end_y, num=50)

    if dim == 3:
        point_z = point_x 
        # Ref: https://stackoverflow.com/a/12864490/11825645
        points = np.array(list(zip(*(x.flat for x in np.meshgrid(point_x, point_y, point_z)) )))
    else:
        points = np.array(list(zip(*(x.flat for x in np.meshgrid(point_x, point_y)) )))

    sample_idx = np.random.choice(len(points), num_train_samples + num_test_samples, replace=False)
    samples = points[sample_idx]

    np.random.shuffle(samples)
    train_target_position = samples[:num_train_samples]
    test_target_position = samples[num_train_samples: num_train_samples+num_test_samples]
    return train_target_position, test_target_position

# ref: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def train_loop(dataloader, test_dataloader, model, optimizer, n_epoch, tb_writer):
    model.train()
    size_of_batches = len(dataloader)
    timestep.add_log(f"{green_head}========== training =========={color_tail}") 
    for n_batch, batch_of_target_pos in enumerate(dataloader):
        timestep.add_log(f"{red_head}==================================================== n_batch = {n_batch} ================================{color_tail}") 
        loss = model(batch_of_target_pos)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = n_batch + n_epoch * size_of_batches
        if n_batch % 1 == 0:
            loss, current = loss.item(), n_batch 
            tb_writer.add_scalar('Loss/train', loss, count)
            timestep.add_log(f"{yellow_head}loss: {loss:>7f}[{current:>5d}/{size_of_batches:>5d}]{color_tail}")

        n = 50
        if n_batch % n == 0:
            test_loop(test_dataloader, model, count, tb_writer)

        n_save = 100
        if n_batch % n_save == 0:
            from time import localtime, strftime
            now = strftime("%Y-%m-%d-%H:%M:%S", localtime())
            timestep.add_log(f"{green_head}==================================================== save model !================================{color_tail}") 
            torch.save(model.state_dict(),
                       f"./saved_models/{now}-bottle-flip-robot-batch-count{count}.pth")
            



def test_loop(dataloader, model, n_epoch, tb_writer):
    num_batches = len(dataloader)
    model.eval()
    test_loss =  0

    timestep.add_log(f"{cyan_head}========== testing =========={color_tail}") 
    with torch.no_grad():
        for n_batch, batch_of_target_pos in enumerate(dataloader):
            loss = model(batch_of_target_pos)
            for l in loss:
                test_loss += l.item()

        test_loss /= num_batches 
        # test_loss = 0
        timestep.add_log(f"{cyan_head}Test Error: \n Avg loss: {test_loss:>8f} \n{color_tail}")
        tb_writer.add_scalar('Loss/test', test_loss, n_epoch)

    model.train()

def main():
    global timestep 
    dim = args.dim

    # ----------------------------------------------
    simulator_base = sph.Exec.SimulatorBase()
    simulator_base.init(sceneFile=os.path.abspath(args.scene), 
                        useGui=not args.no_gui, 
                        initialPause=not args.no_initial_pause,\
                        useCache=not args.no_cache, 
                        stopAt=args.stopAt, 
                        stateFile=os.path.abspath(args.state), \
                        loadFluidPos = args.load_fluid_pos,
                        outputDir=os.path.abspath(args.output_dir), 
                        param=args.param)
    gui = sph.GUI.Simulator_GUI_imgui(simulator_base)
    simulator_base.setGui(gui)

    simulator_base.initSimulationWithDeferredInit()
    
    sim = sph.Simulation.getCurrent()
    timestep = sim.getTimeStep()
    # ---------------------------------------------
    train_target_positions, test_target_positions = generate_dataset(dim, args.num_train, args.num_test)
    train_dataloader = DataLoader(targetPointDataset(train_target_positions), 
                                 batch_size=args.batchsize, shuffle=True)
    test_dataloader = DataLoader(targetPointDataset(test_target_positions), 
                                 batch_size=args.batchsize, shuffle=True)

    # ----------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestep.add_log(f"{green_head}args = {args}{color_tail}")
    timestep.add_log(f"{yellow_head}Using {device} device{color_tail}")

    model = Robot(dim=dim, simulator=simulator_base).to(device)

    # ----------------------------------------------
    if args.model_path == '': # train model 
        tb_writer = SummaryWriter()
        epochs = args.epoch
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        epoch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

        from time import localtime, strftime
        now = strftime("%Y-%m-%d-%H:%M:%S", localtime())
        for n_epoch in range(epochs):
            timestep.add_log(f"{green_head}Epoch {n_epoch+1}\n-------------------------------{color_tail}")
            train_loop(train_dataloader, test_dataloader, model, optimizer, n_epoch, tb_writer)
            torch.save(model.state_dict(),
                       f"./saved_models/{now}-bottle-flip-robot-epoch{n_epoch}.pth")
            # test_loop(test_dataloader, model, n_epoch, tb_writer)
            epoch_lr_scheduler.step()

        timestep.add_log("Done!")
        tb_writer.close()
        simulator_base.cleanup()
    else:
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        print_green("======== test ==========")
        print_green(f" test target pos = {args.test_target}")
        loss = model(torch.Tensor([args.test_target]))
        print_green(f" test loss = {loss}")

main()
