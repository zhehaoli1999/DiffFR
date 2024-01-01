import sys
sys.path.append("./")
sys.path.append("../")
from research.utils import * 
import numpy as np
import os
from pathlib import Path

current_file_path = os.path.abspath(__file__)
revision_root_directory = os.path.dirname(current_file_path) # revision dir
project_root_dir = Path(revision_root_directory).parent.parent
print_green(f"project_root_dir = {project_root_dir}")
print_green(f"revision_root_directory = {revision_root_directory}")

# state_file = f"{project_root_dir}/data/state/selectedStateAsia/diff-dambreak-bunny/state_130.bin"

# os.system("conda activate splishsplash")

for i in range(0, 5):
    # scene_file = f"{revision_root_directory}/scenes/exp2/diff-dambreak-bunny-{i}.json"
    scene_file = f"{revision_root_directory}/scenes/exp2/diff-bottle-model-{i}.json"
    state_file = f"{revision_root_directory}/states/exp2/diff-bottle-flip-{i}/state.bin"

    output_dir = f"{revision_root_directory}/outputs/exp2-fix/diff-bottle-model-{i}"
    # command = f"python {project_root_dir}/pySPlisHSPlasH/research/gradient-based-optimize.py \
                # --scene={scene_file} \
                # --load-fluid-pos-and-vel \
                # --state-file={state_file} \
                # --optimizer='adam' \
                # --lr-v=0.1 \
                # --lr-omega=0.1 \
                # --no-gui\
                # --stopAt=10000\
                # --maxIter=100\
                # --output-dir={output_dir}"

    command = f"python {project_root_dir}/pySPlisHSPlasH/research/gradient-based-optimize.py \
                --scene={scene_file}\
                --load-fluid-pos \
                --state-file={state_file} \
                --optimizer='sgd' \
                --lr-v=1.0 \
                --lr-omega=1.0 \
                --stopAt=10000\
                --maxIter=80\
                --no-initial-pause\
                --no-gui\
                --normalizeGrad\
                --positionWeight=0.1\
                --rotationWeight=1.0\
                --patience=3\
                --output-dir={output_dir}\
                --taskType='bottle-flip'"

    os.system(command)
    


