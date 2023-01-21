python training_baseline_policy.py --env_id "CheeseMaze-v0" --k 1 --training_episodes 5000 --decaying_rate_qlearning 0.002 --beta 15
python training_baseline_policy.py --env_id "Tiger-v0" --k 1 --training_episodes 5000 --decaying_rate_qlearning 0.002 --beta 0.05
python training_baseline_policy.py --env_id "Voicemail-v0" --k 1 --training_episodes 5000 --decaying_rate_qlearning 0.002 --beta 0.3

python training_baseline_policy.py --env_id "Tiger-v0" --k 0 --training_episodes 5000 --decaying_rate_qlearning 0.002 --beta 0.05
python training_baseline_policy.py --env_id "Voicemail-v0" --k 0 --training_episodes 5000 --decaying_rate_qlearning 0.002 --beta 0.3

python main.py -c 8 -s $(seq -s \  1 500) -p data/CheeseMaze-v0_k-1.pkl
python main.py -c 8 -s $(seq -s \  1 500) -p data/Tiger-v0_k-1.pkl
python main.py -c 8 -s $(seq -s \  1 500) -p data/Voicemail-v0_k-1.pkl

python main.py -c 8 -s $(seq -s \  1 500) -p data/Tiger-v0_k-0.pkl --target_ks 0 1
python main.py -c 8 -s $(seq -s \  1 500) -p data/Voicemail-v0_k-0.pkl --target_ks 0 1

