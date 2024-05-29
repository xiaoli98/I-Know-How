ENV="u-turn-v0"
PROJECT_NAME="sb3_sac_master_uturn-v0"
RUN_NAME="subpolicies_MIR"
ALGO="sac_master.py"

tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles  --lr=5e-5 --batch-size=8192 --tau=0.1 --ent-coef="auto_1" " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles  --lr=5e-5 --batch-size=4096 --tau=0.9 --ent-coef="auto_0.5" " C-m
tmux split-window -h -p 80

tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles   --lr=5e-5 --batch-size=4096 --tau=0.9 --ent-coef="auto_0.5" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles   --lr=5e-5 --batch-size=4096 --tau=0.1 --ent-coef="auto_1" " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles   --lr=5e-5 --batch-size=8192 --tau=0.9 --ent-coef="auto_0.5" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles   --lr=5e-5 --batch-size=8192 --tau=0.2 --ent-coef="auto_1" " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles   --lr=5e-5 --batch-size=2048 --tau=0.9 --ent-coef="auto_0.5" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles  --batch-size=2048 --lr=1e-5 " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles  --batch-size=4096 --lr=1e-5 " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles --batch-size=2048 --lr=5e-6 --tau=0.9 --ent-coef="auto_1" " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles  --batch-size=4096 --lr=5e-6 " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles --batch-size=2048 --lr=1e-6  " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles --batch-size=4096 --lr=5e-6  --tau=0.9 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles   " C-m
tmux send-keys "python3 sac_master.py --env=$ENV --ncpu=16 --project-name=$PROJECT_NAME --run-name=$RUN_NAME --ntags random_spawn 5_vehicles   " C-m
