ENV="racetrack-v0"
PROJECT_NAME="sb3_sac_master_5vehicles"
SUBPOLICIES="MIRIN"
RUN_NAME="subpolicies_${SUBPOLICIES}"

RANDOM_SPAWN=true

if $RANDOM_SPAWN; then
    RANDOM_SPAWN_TAG="random_spawn"
else
    RANDOM_SPAWN_TAG=""
fi
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
# tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
# tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
# tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
# tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
# tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
# tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
# tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
# tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --subpolicies=$SUBPOLICIES --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG 5_vehicles non_det_subp negative_weighting   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
