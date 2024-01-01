from taskutils import *
from pathlib import Path
import os

lbfgs_iter_time = 50
ng_iter_time = 100
sgd_iter_time = ng_iter_time

state_dict = {
              "diff-dambreak-bunny": "diff-dambreak-bunny/state_130",
              "diff-dambreak-bunny-difficult": "diff-dambreak-bunny/state_0.40014",
              "diff-bottle-model": "diff-bottle-model/state_54",
              "diff-bottle-model-360degree": "diff-bottle-model/state_54",
              "diff-bottle-model-360degree-multipath": "diff-bottle-model/state_54",
              "diff-bottle-model-2-360degree": "diff-bottle-model/state_54",
              "diff-bottle-bunny-360degree": "diff-bottle-bunny/state_8.00204",
              "diff-stone-skipping":"diff-stone-skipping/state_18.402088",
              "diff-high-diving-armadillo": None,
              "diff-high-diving-armadillo-difficult": "diff-high-diving/state_1.355999",
              "diff-high-diving-duck": "diff-high-diving/state_98"
              }

def get_ES_task(scene, optimizer, position_weight=1.0, rotation_weight=1.0):
    task_type = "dambreak-bunny"
    if 'bottle' in scene:
        task_type = "bottle-flip"
    elif 'stone' in scene:
        task_type = "stone-skipping"
    elif 'high' in scene:
        task_type = "high-diving"

    return Task(py="opt-ng", scene=scene,
                 state=state_dict[scene],
                 optimizer=optimizer,
                 task_type=task_type,
                 position_weight=position_weight,
                 rotation_weight=rotation_weight,
                 maxIter= ng_iter_time)

def get_sgd_task(scene, lr_v=1., lr_omega=1., patience=5, gradient_mode=1, normalizeGrad=False,
                 position_weight=1.0, rotation_weight=1.0):
    task_type = "dambreak-bunny"
    if 'bottle' in scene:
        task_type = "bottle-flip"
    elif 'stone' in scene:
        task_type = "stone-skipping"
    elif 'high' in scene:
        task_type = "high-diving"

    return  Task(py="diff-sgd-new", scene=scene, 
                 state=state_dict[scene],
                 optimizer='sgd',
                 task_type=task_type,
                 lr_v = lr_v, 
                 lr_omega = lr_omega,
                 gradient_mode=gradient_mode,
                 patience=patience,
                 normalize_grad=normalizeGrad,
                 position_weight=position_weight,
                 rotation_weight=rotation_weight,
                 maxIter= sgd_iter_time)


task_queue = [
# --------------------- diff-high-diving -----------------------
          #  get_sgd_task("diff-high-diving-armadillo", lr_omega = 4, patience=3),
            # get_sgd_task("diff-high-diving-armadillo", lr_omega = 2, patience=3),
            # get_ES_task("diff-high-diving-armadillo", "cma"), 
            # get_ES_task("diff-high-diving-armadillo", "1+1es"), 

# # --------------------- diff-stone-skipping -----------------------
            # get_sgd_task("diff-stone-skipping", lr_v=4, patience=5),
            # get_sgd_task("diff-stone-skipping", lr_v=5, patience=2),
          #  get_sgd_task("diff-stone-skipping", lr_v=8, patience=2),
            
# # --------------------- diff-bottle-model -----------------------
          #  get_sgd_task("diff-bottle-model", lr_v=1, position_weight=0.1, normalizeGrad=True),
            # get_ES_task("diff-bottle-model", "cma", position_weight=0.1), 
            # get_ES_task("diff-bottle-model", "1+1es", position_weight=0.1), 

# # -------------------diff-bottle-model-360degree ----------------
          #  get_sgd_task("diff-bottle-model-360degree", lr_v=1, position_weight=0.1, normalizeGrad=True),
            # get_ES_task("diff-bottle-model-360degree", "cma", position_weight=0.1), 
            # get_ES_task("diff-bottle-model-360degree", "1+1es", position_weight=0.1), 

# # -------------------diff-bottle-model-2-360degree ----------------
          #   get_sgd_task("diff-bottle-model-2-360degree", lr_v=1, position_weight=0.1, normalizeGrad=True), 
            # get_ES_task("diff-bottle-model-2-360degree", "cma", position_weight=0.1), 
            # get_ES_task("diff-bottle-model-2-360degree", "1+1es", position_weight=0.1), 

# # -------------------diff-bottle-model-360degree-multipath ----------------
           # get_sgd_task("diff-bottle-model-360degree-multipath", lr_v=1, position_weight=0.1, normalizeGrad=True),

# ------------------ diff-dambreak-bunny ==================
            # get_sgd_task("diff-dambreak-bunny", lr_v=2, lr_omega=2, position_weight=1.3),
            get_ES_task("diff-bottle-model", "cma", position_weight=1.0, rotation_weight=1.0), 
            get_ES_task("diff-bottle-model", "1+1es", position_weight=1.0, rotation_weight=1.0), 
            # get_ES_task("diff-high-diving-duck", "cma", position_weight=1.0, rotation_weight=1.0), 
            # get_ES_task("diff-high-diving-duck", "1+1es", position_weight=1.0, rotation_weight=1.0), 
            # get_ES_task("diff-dambreak-bunny", "1+1es", position_weight=1.3), 

            # get_sgd_task("diff-dambreak-bunny", lr_v=1, lr_omega=4, position_weight=1),
            # get_sgd_task("diff-dambreak-bunny", lr_v=2, lr_omega=2, position_weight=1),
            # get_sgd_task("diff-dambreak-bunny", lr_v=1, lr_omega=2, position_weight=1),
            # get_sgd_task("diff-dambreak-bunny", lr_v=0.5, lr_omega=2, position_weight=1),
            # get_ES_task("diff-dambreak-bunny", "cma", position_weight=1), 
            # get_ES_task("diff-dambreak-bunny", "1+1es", position_weight=1) 

# # ---------------- diff-bottle-bunny-360degree ----------
            # get_sgd_task("diff-bottle-bunny-360degree", lr_v = 0.5, lr_omega=0.2, patience=3, position_weight=0.1, normalizeGrad=True),
            # get_sgd_task("diff-bottle-bunny-360degree", lr_v = 0.5, lr_omega=0.5, patience=5, position_weight=0.1, normalizeGrad=True),
            # get_sgd_task("diff-bottle-bunny-360degree", lr_v = 0.5, lr_omega=1, patience=5, position_weight=0.1, normalizeGrad=True),
            # get_ES_task("diff-bottle-bunny-360degree", "cma", position_weight=0.1), 
            # get_ES_task("diff-bottle-bunny-360degree", "1+1es", position_weight=0.1), 

# -----------------------------------------------------
# ---------------- diff-dambreak-bunny-difficult  -------------------------------
           # get_sgd_task("diff-dambreak-bunny-difficult", lr_v=1, lr_omega=1, position_weight=1),
            # get_sgd_task("diff-dambreak-bunny-difficult", lr_v=1, lr_omega=2, position_weight=1),
            # get_sgd_task("diff-dambreak-bunny-difficult", lr_v=1, lr_omega=4, position_weight=1),

# ---------------- diff-high-diving-armadillo-difficult  -------------------------------
            #get_sgd_task("diff-high-diving-armadillo-difficult", lr_omega=1, position_weight=1),
            #get_sgd_task("diff-high-diving-armadillo-difficult", lr_omega=4, position_weight=1),

]

run(task_queue, src_path= Path(os.getcwd()).parent.parent.absolute())
