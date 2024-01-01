import pysplishsplash as sph
import sys
sys.path.append("./")
sys.path.append("../")
from argutils import * 
from research.utils import * 
import numpy as np 
from numpy.linalg import norm

# ==================================================================
# 
#               3. timestep 
#  current state -----------> next state (record force & our gradient)
#     ^   | 1. perturb delta_v                       | --------------> 4. compute finite diffrence gradient
#     |   |-----------------> new state (record newforce)
#     |                            |
#     |   2. retore state          |
#     |----------------------------|
#
# ==================================================================

import os
from pathlib import Path
current_file_path = os.path.abspath(__file__)
revision_root_directory = os.path.dirname(current_file_path) # revision dir
project_root_dir = Path(revision_root_directory).parent.parent
print_green(f"project_root_dir = {project_root_dir}")
print_green(f"revision_root_directory = {revision_root_directory}")

# scene_file = f"{revision_root_directory}/scenes/diff-dambreak-bunny.json"
# state_file = f"{project_root_dir}/data/state/selectedStateAsia/diff-dambreak-bunny/state_130.bin"

scene_file = f"{revision_root_directory}/scenes/diff-stone-skipping-r0.015.json"
state_file = f"{revision_root_directory}/states/diff-stone-skipping-r0.015/state.bin"

# First, load scene of bunny water rafting 
base = sph.Exec.SimulatorBase()
base.init(sceneFile=scene_file, useGui=True, initialPause=True,
          useCache=True, stopAt=-1, stateFile=state_file,
          loadFluidPos=False, loadFluidPosAndVel=True,
          outputDir=f"{revision_root_directory}/outputs")
gui = sph.GUI.Simulator_GUI_imgui(base)
base.setGui(gui)

# base.initSimulation()
base.initSimulationWithDeferredInit()
sim = sph.Simulation.getCurrent()

diffdfsph_timestep = sim.getTimeStep()
# timestep.add_log(f"{green_head}{args}{color_tail}")
diffdfsph_timestep.add_log(
    f"{green_head} gradient mode = {sim.getGradientMode()} (0 for complete, 1 for incomplete, 2 for rigid) {color_tail}")

state_dir = f"{revision_root_directory}/states/finite-difference-tmp/"

def save_current_state():
    tmp_state_dir = f"{state_dir}"
    tmp_state_file = base.saveState(tmp_state_dir)
    return f"{tmp_state_file}.bin"

def restore_state(state_file):
    base.loadStateWithRigidExisted(state_file)
    # print_green(f"restore state from file {state_file}")

# --------------------------------------------------------------
# delta_list = [1e-4, 1e-6, 1e-8, 1e-10]
# delta = 1e-100
delta = 0
finite_diff_grad_list = []
our_grad_list = []
grad_error_list = []

all_force_list = []
new_force_list = []

tm = sph.TimeManager.getCurrent()

def main():
    not_end_trajectory = True
    count_step = 0

    while not_end_trajectory:
        state_file_path = save_current_state()
        # print_yellow(state_file_path)

        bm = sim.getBoundaryModel(1)
        rb = bm.getRigidBodyObject()
        old_v = rb.getVelocity()

        # ----------- velocity perturbation ---------------
        new_force_list.clear()
        for i in range(3):
            new_v = old_v 
            new_v[i] += delta 

            rb.setVelocity(new_v)
            
            # set new velocity of rigid body 
            base.singleTimeStep() # This leads to recursion problem!
            # gui.show()
            # get new forces 
            new_force = bm.getForce() 

            restore_state(state_file_path)

            new_force_list.append(new_force)
    
        # ----------- step forward ---------------
        base.singleTimeStep()

        # ----------- compute finite diff grad ---------------
        force = bm.getForce()
        # compute finite difference gradient 
        finite_diff_grad = np.zeros((3,3))
        for i, f in enumerate(new_force_list):
            finite_diff_grad_i = (f - force) / delta 
            finite_diff_grad[i, :] = finite_diff_grad_i # need to check row-major or column-major
        all_force_list.append([new_force_list[:], force])

        # ----------- get our grad & compute error ---------------
        rb_grad_manager = base.getRigidBodyGradientManager()
        rb_idx = 1
        our_grad = rb_grad_manager.get_grad_net_force_to_vn(rb_idx, rb_idx)

        # compute error 
        error = norm(our_grad - finite_diff_grad) / (norm(finite_diff_grad) + 1e-10 )

        if count_step % 10 == 0:
            t = tm.getTime()
            print(f" --------------- [time {t}] --------------------") 
            print_cyan(f"new_force = {new_force_list}")
            print_cyan(f"force = {force.transpose()}")
            print_cyan(f"finite_diff_grad: {finite_diff_grad}")
            print_green(f"our_grad: {our_grad}")
            print_yellow(f"||gradient error|| = {error}")

        grad_error_list.append(error)
        finite_diff_grad_list.append(finite_diff_grad)
        our_grad_list.append(our_grad)

        if diffdfsph_timestep.is_trajectory_finish_callback():
            not_end_trajectory = False
            break 

        count_step += 1

    # ------------- finially, save result ----------------------------------
    import pickle as pk 
    result_dir = f"{revision_root_directory}/results/finite-difference"
    with open(f"{result_dir}/saved_data-stone-skipping-delta{delta}.pk", 'wb') as file:
        # Serialize and save the object using pickle.dump()
        data = {"grad_error_list": grad_error_list,
                "all_force_list": all_force_list,
                "finite_diff_grad_list": finite_diff_grad_list, 
                "our_grad_list": our_grad_list}
        pk.dump(data, file)

    base.cleanup()

main()
