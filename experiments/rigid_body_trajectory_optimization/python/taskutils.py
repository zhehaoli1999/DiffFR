import subprocess
from collections import namedtuple
from datetime import datetime
import os
from utils import *

now = datetime.now()
my_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # dd/mm/YY H:M:S

# Task = namedtuple('Task', ['py', 'scene', 'state', 'load_fluid_method', 'output_dir', 'lr_v', 'lr_omega', 'momentum', 'weight', 'patience', 'maxIter'])

optimizer_dict = {"sgd": "diff-sgd-new", "lbfgs": "diff-lbfgs-new", "cma": "opt-ng", "1+1es": "opt-ng"}
method_string_dict = {"1+1es": "-oneplusone"}

class Task:
    def __init__(self, py, scene, state=None, load_fluid_method='load-fluid-pos',
                 output_dir=None, lr_v=1.0,
                 lr_omega=1.0, momentum=0.5, 
                 position_weight=1.0,
                 rotation_weight=1.0,
                 penalty_weight=1.0,
                 patience=5, maxIter=150,
                 optimizer="sgd", 
                 task_type="bottle-flip", # bottle-flip, stone-skipping, high-diving
                 full_grad=False,
                 normalize_grad=True,
                 gradient_mode = 1, # 0 for complete, 1 for incomplete, 2 for rigid-only
                 other_args=[],
                 use_gui=False) -> None:

        self.py = py 
        self.scene = scene 
        self.state = state
        self.load_fluid_method = load_fluid_method 
        if task_type in ["dambreak-bunny"]:
            self.load_fluid_method = "load-fluid-pos-and-vel"
        self.output_dir = output_dir 
        self.lr_v = lr_v
        self.lr_omega= lr_omega
        self.momentum = momentum 

        self.position_weight = position_weight 
        self.rotation_weight = rotation_weight 
        self.penalty_weight = penalty_weight 

        self.patience = patience
        self.maxIter = maxIter 
        self.optimizer = optimizer
        self.task_type = task_type
        self.other_args = other_args
        self.full_grad = full_grad
        self.normalize_grad = normalize_grad
        self.gradient_mode = gradient_mode 
        self.use_gui = use_gui
        
        method_string = method_string_dict.get(optimizer)
        if method_string is not None:
            self.method_string = method_string
        else:
            self.method_string = '-' + optimizer

        recommandFile = optimizer_dict.get(optimizer)
        if recommandFile is not None and recommandFile != py:
            print_yellow(f"Warning: recommand using {recommandFile}.py for {optimizer}-type optimization")

def run(task_queue, src_path):
    for idx, t in enumerate(task_queue):
        t: Task
        output_dir =f"{src_path}/Results/{t.scene}{t.method_string}-{my_time}-[{idx}]"
        if t.output_dir is not None:
            output_dir = t.output_dir

        os.makedirs(output_dir, exist_ok=False)
        
        command = f"python {src_path}/pySPlisHSPlasH/research/{t.py}.py \
                        --scene={src_path}/data/Scenes/selectedScenesAsia/{t.scene}.json \
                        --{t.load_fluid_method} \
                        --output-dir={output_dir} \
                        --positionWeight={t.position_weight} \
                        --rotationWeight={t.rotation_weight} \
                        --penaltyWeight={t.penalty_weight} \
                        --maxIter={t.maxIter} "

        if not t.use_gui:
            command += f"--no-gui --stopAt=1000 "

        if t.optimizer == 'sgd':
            command += f"--lr-v={t.lr_v} \
                        --lr-omega={t.lr_omega} \
                        --momentum={t.momentum} \
                        --patience={t.patience} \
                        --gradientMode={t.gradient_mode}   "
            if t.normalize_grad:
                command += "--normalizeGrad "
            
            if t.full_grad:
                command += "--fullDerivative "

        elif t.optimizer == 'cma':
            command += f"--optimizer=CMAES "
        
        elif t.optimizer == '1+1es':
            command += f"--optimizer=1+1ES "

        # elif t.optimizer == 'lbfgs':
            # command += f"--weight={t.weight} \
                        # --maxIter={t.maxIter} "
        else:
            print_red(f"Unknown optimizer {t.optimizer}")
            continue

        if t.task_type is not None:
            if t.task_type in ['bottle-flip' ,'stone-skipping', 'high-diving', 'dambreak-bunny']:
                command += f"--taskType={t.task_type} "
            else:
                print_red(f"Unknown optimization mode {t.task_type}")
                continue

        if t.state is not None:
            command += f"--state={src_path}/data/state/selectedStateAsia/{t.state}.bin "  
           

        for arg in t.other_args:
            command += arg

        os.system(command)

class StoneSkippingTask(Task):
    def __init__(self, output_dir=None, lr_v=1.0,
                 lr_omega=1.0, momentum=0.5, 
                 position_weight=1,
                 rotation_weight=1,
                 penalty_weight=1,
                 patience=5, maxIter=150,
                 optimizer="sgd", 
                 full_grad=False,
                 normalize_grad=False,
                 gradient_mode = 1, # 0 for complete, 1 for incomplete, 2 for rigid-only
                 other_args=[]) -> None:
        
        py = None
        if optimizer == 'sgd':
            py = 'diff-sgd-new'
        else:
            py = 'opt-ng'

        super().__init__(py=py, scene="diff-stone-skipping", state="diff-stone-skipping/state_0.546885", 
                         load_fluid_method="load-fluid-pos", 
                         output_dir=output_dir, 
                         lr_v=lr_v,
                         lr_omega=lr_omega,
                         momentum=momentum,
                         position_weight=position_weight,
                         rotation_weight=rotation_weight,
                         penalty_weight=penalty_weight,
                         patience=patience,
                         maxIter=maxIter,
                         optimizer=optimizer,
                         task_type='stone-skipping',
                         full_grad=full_grad,
                         normalize_grad=normalize_grad,
                         gradient_mode=gradient_mode,
                         other_args=other_args)

