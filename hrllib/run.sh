python test_rllib_ppo.py 0 2 2 10 1>2210.04ppo 2>b&
python test_rllib_ppo.py 1 2 5 10 1>2510.04ppo 2>b &
python test_rllib_ppo.py 2 5 5 5 1>555.04ppo 2>b &
python test_rllib_ppo.py 3 5 5 15 1>5515.04ppo 2>b &
python test_rllib_ppo.py 4 10 5 10 1>10510.04ppo 2>b &
python test_rllib_ppo.py 5 5 10 15 1>51015.04ppo 2>b &
python test_rllib_ppo.py 6 10 10 20 1>101020.04ppo 2>b &
python test_rllib_ppo.py 7 20 5 30 1>20530.04ppo 2>b &

python test_rllib_ppo.py 0 2 2 10 1>2210.06ppo 2>b&
python test_rllib_ppo.py 1 2 5 10 1>2510.06ppo 2>b &
python test_rllib_ppo.py 2 5 5 5 1>555.06ppo 2>b &
python test_rllib_ppo.py 3 5 5 15 1>5515.06ppo 2>b &
python test_rllib_ppo.py 4 10 5 10 1>10510.06ppo 2>b &
python test_rllib_ppo.py 5 5 10 15 1>51015.06ppo 2>b &
python test_rllib_ppo.py 6 10 10 20 1>101020.06ppo 2>b &
python test_rllib_ppo.py 7 20 5 30 1>20530.06ppo 2>b &

python test_rllib_ppo.py 0 2 2 10 1>2210.10ppo 2>b&
python test_rllib_ppo.py 1 2 5 10 1>2510.10ppo 2>b &
python test_rllib_ppo.py 2 5 5 5 1>555.10ppo 2>b &
python test_rllib_ppo.py 3 5 5 15 1>5515.10ppo 2>b &
python test_rllib_ppo.py 4 10 5 10 1>10510.10ppo 2>b &
python test_rllib_ppo.py 5 5 10 15 1>51015.10ppo 2>b &
python test_rllib_ppo.py 6 10 10 20 1>101020.10ppo 2>b &
python test_rllib_ppo.py 7 20 5 30 1>20530.10ppo 2>b &

python test_rllib.py 0 2 2 10 1>2210.04pg 2>b&
python test_rllib.py 1 2 5 10 1>2510.04pg 2>b &
python test_rllib.py 2 5 5 5 1>555.04pg 2>b &
python test_rllib.py 3 5 5 15 1>5515.04pg 2>b &
python test_rllib.py 4 10 5 10 1>10510.04pg 2>b &
python test_rllib.py 5 5 10 15 1>51015.04pg 2>b &
python test_rllib.py 6 10 10 20 1>101020.04pg 2>b &
python test_rllib.py 7 20 5 30 1>20530.04pg 2>b &

python test_rllib.py 0 2 2 10 1>2210.06pg 2>b&
python test_rllib.py 1 2 5 10 1>2510.06pg 2>b &
python test_rllib.py 2 5 5 5 1>555.06pg 2>b &
python test_rllib.py 3 5 5 15 1>5515.06pg 2>b &
python test_rllib.py 4 10 5 10 1>10510.06pg 2>b &
python test_rllib.py 5 5 10 15 1>51015.06pg 2>b &
python test_rllib.py 6 10 10 20 1>101020.06pg 2>b &
python test_rllib.py 7 20 5 30 1>20530.06pg 2>b &

python test_rllib.py 0 2 2 10 1>2210.10pg 2>b&
python test_rllib.py 1 2 5 10 1>2510.10pg 2>b &
python test_rllib.py 2 5 5 5 1>555.10pg 2>b &
python test_rllib.py 3 5 5 15 1>5515.10pg 2>b &
python test_rllib.py 4 10 5 10 1>10510.10pg 2>b &
python test_rllib.py 5 5 10 15 1>51015.10pg 2>b &
python test_rllib.py 6 10 10 20 1>101020.10pg 2>b &
python test_rllib.py 7 20 5 30 1>20530.10pg 2>b &

cat 2210.04pg | grep episode_reward_mean > 2210.04p
cat 2510.04pg | grep episode_reward_mean > 2510.04p
cat 555.04pg | grep episode_reward_mean > 555.04p
cat 5515.04pg | grep episode_reward_mean > 5515.04p
cat 10510.04pg | grep episode_reward_mean > 10510.04p
cat 51015.04pg | grep episode_reward_mean > 51015.04p
cat 101020.04pg | grep episode_reward_mean > 101020.04p
cat 20530.04pg | grep episode_reward_mean > 20530.04p

cat 2210.06pg | grep episode_reward_mean > 2210.06p
cat 2510.06pg | grep episode_reward_mean > 2510.06p
cat 555.06pg | grep episode_reward_mean > 555.06p
cat 5515.06pg | grep episode_reward_mean > 5515.06p
cat 10510.06pg | grep episode_reward_mean > 10510.06p
cat 51015.06pg | grep episode_reward_mean > 51015.06p
cat 101020.06pg | grep episode_reward_mean > 101020.06p
cat 20530.06pg | grep episode_reward_mean > 20530.06p

cat 2210.04ppo | grep episode_reward_mean > 2210.04o
cat 2510.04ppo | grep episode_reward_mean > 2510.04o
cat 555.04ppo | grep episode_reward_mean > 555.04o
cat 5515.04ppo | grep episode_reward_mean > 5515.04o
cat 10510.04ppo | grep episode_reward_mean > 10510.04o
cat 51015.04ppo | grep episode_reward_mean > 51015.04o
cat 101020.04ppo | grep episode_reward_mean > 101020.04o
cat 20530.04ppo | grep episode_reward_mean > 20530.04o

cat 2210.06ppo | grep episode_reward_mean > 2210.06o
cat 2510.06ppo | grep episode_reward_mean > 2510.06o
cat 555.06ppo | grep episode_reward_mean > 555.06o
cat 5515.06ppo | grep episode_reward_mean > 5515.06o
cat 10510.06ppo | grep episode_reward_mean > 10510.06o
cat 51015.06ppo | grep episode_reward_mean > 51015.06o
cat 101020.06ppo | grep episode_reward_mean > 101020.06o
cat 20530.06ppo | grep episode_reward_mean > 20530.06o

cat 2210.10ppo | grep episode_reward_mean > 2210.10o
cat 2510.10ppo | grep episode_reward_mean > 2510.10o
cat 555.10ppo | grep episode_reward_mean > 555.10o
cat 5515.10ppo | grep episode_reward_mean > 5515.10o
cat 10510.10ppo | grep episode_reward_mean > 10510.10o
cat 51015.10ppo | grep episode_reward_mean > 51015.10o
cat 101020.10ppo | grep episode_reward_mean > 101020.10o
cat 20530.10ppo | grep episode_reward_mean > 20530.10o

