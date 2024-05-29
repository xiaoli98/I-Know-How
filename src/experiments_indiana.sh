SAC_ENV="indiana-v0"
SAC_PROJECT_NAME="sb3_sac_indiana"
tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac.py --env=$SAC_ENV --ncpu=16 --project-name=$SAC_PROJECT_NAME  --lr=5e-5 --batch-size=8192 --tau=0.1 --ent-coef="auto_0.5" --gamma=0.65" C-m
# tmux send-keys "python3 sac.py --env=$SAC_ENV --ncpu=16 --project-name=$SAC_PROJECT_NAME  --lr=2e-5 --batch-size=2048 --tau=0.2 --ent-coef="auto_1" --gamma=0.7" C-m
# tmux send-keys "python3 sac.py --env=$SAC_ENV --ncpu=16 --project-name=$SAC_PROJECT_NAME  " C-m
# tmux send-keys "python3 sac.py --env=$SAC_ENV --ncpu=16 --project-name=$SAC_PROJECT_NAME  " C-m

tmux split-window -h -p 80
tmux send-keys "source ./venv/bin/activate" C-m
tmux send-keys "python3 sac.py --env=$SAC_ENV --ncpu=16 --project-name=$SAC_PROJECT_NAME  " C-m
# tmux send-keys "python3 sac.py --env=$SAC_ENV --ncpu=16 --project-name=$SAC_PROJECT_NAME  " C-m
# tmux send-keys "python3 sac.py --env=$SAC_ENV --ncpu=16 --project-name=$SAC_PROJECT_NAME  --lr=5e-5 --batch-size=8192 --tau=0.1 --ent-coef="auto_0.5" --gamma=0.65" C-m
# tmux send-keys "python3 sac.py --env=$SAC_ENV --ncpu=16 --project-name=$SAC_PROJECT_NAME  --lr=5e-5 --batch-size=4096 --tau=0.9 --ent-coef="auto_0.5" --gamma=0.65" C-m
