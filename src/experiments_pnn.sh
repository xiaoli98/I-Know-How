ENV="racetrack-v0"
PROJECT_NAME="sac_pnn_racetrack-v0"

RANDOM_SPAWN=true

if $RANDOM_SPAWN; then
    RANDOM_SPAWN_TAG="random_spawn"
else
    RANDOM_SPAWN_TAG=""
fi

tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_pnn.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME"  --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_pnn.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME"  --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_pnn.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME"  --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_pnn.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME"  --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_pnn.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME"  --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_pnn.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME"  --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_pnn.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME"  --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac_pnn.py --env=$ENV --ncpu=16 --project-name="$PROJECT_NAME"  --random_spawn=$RANDOM_SPAWN  --ntags $RANDOM_SPAWN_TAG " C-m

