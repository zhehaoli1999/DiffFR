import pysplishsplash as sph 
import sys

from torch.nn.modules.fold import F
sys.path.append("./")
sys.path.append("../")
from argutils import * 
from research.utils import * 
import numpy as np 
from numpy.linalg import norm
import pickle as pk 

# get root directory
import os
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
root_dir = parent_directory

# Use the stone skipping scene 
# first, create a series scene Json files with a range of resolution
r_list = [0.015, 0.012, 0.009, 0.006, 0.005, 0.004]
# r_list = [0.005]
# r_list = [0.004]
# r_list = [0.015]
# r_list = ["test"]
scene_list = []
for r in r_list:
    scene_list.append(f"diff-stone-skipping-r{r}")
    # scene_list.append(f"test")

print_green(scene_list)

avg_norm_grad_list = []
avg_num_1ring_fluid_particle_list = []

base = sph.Exec.SimulatorBase()

n_repeat = 5
for n in range(n_repeat):
    for i, scene in enumerate(scene_list): 
        # load_fluid_pos_and_vel = False
        # if '0.004' in scene:
            # load_fluid_pos_and_vel = True
        load_fluid_pos_and_vel = True
        print_green(f"load_fluid_pos_and_vel: {load_fluid_pos_and_vel}")

        base.init(sceneFile=f"{root_dir}/scenes/{scene}.json", 
                  useGui =True, initialPause=False, useCache=True,
                  stopAt=-1.0,
                  stateFile=f"{root_dir}/states/{scene}/state.bin",
                  loadFluidPos = not load_fluid_pos_and_vel,
                  # loadFluidPos = False,
                  loadFluidPosAndVel = load_fluid_pos_and_vel, 
                  outputDir = f"{root_dir}/outputs/{scene}"
                  )
        gui = sph.GUI.Simulator_GUI_imgui(base)
        base.setGui(gui)

        base.initSimulationWithDeferredInit()
        # base.initSimulation() # not here this is wrong, need to deferredInit
        sim = sph.Simulation.getCurrent()

        diffdfsph_timestep = sim.getTimeStep()

        not_end_trajectory = True
        step_count = 0 
        sum_norm_gradient = 0.
        sum_num_1ring_fluid_particle = 0

        rb_idx = 1

        grad_list = []
        num_1ring_fluid_particle_list = []

        while not_end_trajectory:
           
            base.singleTimeStep()

            # would like to show gui here 
            # gui.show() # Note: works with single scene, error when multiple scenes

            rb_grad_manager = base.getRigidBodyGradientManager()

            tm = sph.TimeManager.getCurrent()
            t = tm.getTime()

            # is this value cleared at the end of each time step? Yes
            gradient_force_to_v = rb_grad_manager.get_grad_net_force_to_vn(rb_idx, rb_idx)
            grad_norm = norm(gradient_force_to_v)

            num_1ring_fluid_particle = diffdfsph_timestep.get_num_1ring_fluid_particle()
            print_green(f"[round{n}:scene{i}][time {t}] num 1ring = {num_1ring_fluid_particle}, ||grad|| = {grad_norm}")

            grad_list.append(gradient_force_to_v)
            num_1ring_fluid_particle_list.append(num_1ring_fluid_particle)

            n_record_thresh = 100
            if num_1ring_fluid_particle > n_record_thresh:
                sum_norm_gradient += grad_norm
                sum_num_1ring_fluid_particle += num_1ring_fluid_particle
                step_count += 1

            if diffdfsph_timestep.is_trajectory_finish_callback():
                not_end_trajectory = False
                # gui.endShow()
                break 
            
            # break # For Debugging

        if step_count > 0: 
            avg_norm_gradient = sum_norm_gradient / step_count
            avg_num_1ring_fluid_particle = sum_num_1ring_fluid_particle / step_count
        else:
            avg_norm_gradient = 0.
            avg_num_1ring_fluid_particle = 0.

        avg_norm_grad_list.append(avg_norm_gradient)
        avg_num_1ring_fluid_particle_list.append(avg_num_1ring_fluid_particle)
        print_green(f"avg_norm_grad_list = {avg_norm_grad_list}")
        print_green(f"avg_num_1ring_list = {avg_num_1ring_fluid_particle_list}")

        result_dir = f"{root_dir}/results/sensitivity-1ring"

        with open(f"{result_dir}/saved_data-repeat{n}-scene{i}.pk", 'wb') as file:
            # Serialize and save the object using pickle.dump()
            data = {"avg_norm_grad": avg_norm_grad_list,
                    "avg_num_1ring": avg_num_1ring_fluid_particle_list}
            pk.dump(data, file)
        with open(f"{result_dir}/saved_data_detail-repeat{n}-scene{i}.pk", 'wb') as file:
            # Serialize and save the object using pickle.dump()
            data = {"grad": grad_list,
                    "num_1ring": num_1ring_fluid_particle_list}
            pk.dump(data, file)
        print_green("save data to disk.")

        base.cleanup()
        print_green("base cleanup.")

print_green("all scene finished.")
## Finally, plot the curve of ||grad|| w.r.t #{one-ring fluid particles}
# import matplotlib.pyplot as plt 
# plt.plot(avg_num_1ring_fluid_particle_list, avg_norm_grad_list)
# plt.show()

