ENV="lane-centering-v0"
PROJECT_NAME="sb3_sac_lane_centering"

tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" " C-m
# tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --tau=0.005" C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --tau=0.005 --ent-coef="auto_1" " C-m
# tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --lr=1e-5 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --tau=0.005 " C-m
# tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --lr=1e-6 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --lr=1e-5" C-m
# tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --lr=1e-5  " C-m
# tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --lr=1e-5 --tau=0.005" C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --lr=1e-6 --ent-coef="auto_1"" C-m
# tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --lr=1e-6 --tau=0.1" C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --tau=0.005 --ent-coef="auto_1" " C-m
# tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --lr=1e-5 --ent-coef="auto_1" " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --lr=1e-6 --tau=0.005 --ent-coef="auto_1" " C-m
# tmux send-keys "python3 sac.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME" --lr=1e-5 --ent-coef="auto_1" " C-m
