
##cat $1 | grep "EPISODE"| python analysis_episode.log.py $2 > $1.ep.log

cat ../random.task_4.max_step_20.log | grep "EPISODE"| python analysis_episode_res.py 32 > random.task_4.max_step_20.ep.txt
cat ../random.task_4.max_step_40.log | grep "EPISODE"| python analysis_episode_res.py 32 > random.task_4.max_step_40.ep.txt
cat ../random.task_4.max_step_80.log | grep "EPISODE"| python analysis_episode_res.py 32 > random.task_4.max_step_80.ep.txt
cat ../random.task_8.max_step_20.log | grep "EPISODE"| python analysis_episode_res.py 32 > random.task_8.max_step_20.ep.txt
cat ../random.task_8.max_step_40.log | grep "EPISODE"| python analysis_episode_res.py 32 > random.task_8.max_step_40.ep.txt
cat ../random.task_8.max_step_80.log | grep "EPISODE"| python analysis_episode_res.py 32 > random.task_8.max_step_80.ep.txt

cat ../greedy.task_4.max_step_20.log | grep "EPISODE"| python analysis_episode_res.py 32 > greedy.task_4.max_step_20.ep.txt
cat ../greedy.task_4.max_step_40.log | grep "EPISODE"| python analysis_episode_res.py 32 > greedy.task_4.max_step_40.ep.txt
cat ../greedy.task_4.max_step_80.log | grep "EPISODE"| python analysis_episode_res.py 32 > greedy.task_4.max_step_80.ep.txt
cat ../greedy.task_8.max_step_20.log | grep "EPISODE"| python analysis_episode_res.py 32 > greedy.task_8.max_step_20.ep.txt
cat ../greedy.task_8.max_step_40.log | grep "EPISODE"| python analysis_episode_res.py 32 > greedy.task_8.max_step_40.ep.txt
cat ../greedy.task_8.max_step_80.log | grep "EPISODE"| python analysis_episode_res.py 32 > greedy.task_8.max_step_80.ep.txt
