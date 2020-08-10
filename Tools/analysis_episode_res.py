import sys
import numpy as np

result_matrix = []

episode_step = int(sys.argv[1])

for line in sys.stdin:
    try:
        if "EPISODE_" not in line:
            #print line.strip()
            continue
        groups = line.strip().split()
        value_list = []
        for term in groups:
            value = float(term.split(":")[1])
            value_list.append(value)
        result_matrix.append(value_list)
    except:
        continue

# if (len(result_matrix) % episode_step != 0):
#     print "EPISODE NUM ERROR!"

iter_num = len(result_matrix) / episode_step
for i in range(iter_num):
    start = i * episode_step
    end = start + episode_step
    episode_matrix = result_matrix[start:end]
    reward = np.mean([t[0] for t in episode_matrix])
    finish_num = np.mean([t[1] for t in episode_matrix])
    time_mean = np.mean([t[2] for t in episode_matrix])
    time_std = np.mean([t[3] for t in episode_matrix])
    fare_mean = np.mean([t[4] for t in episode_matrix])
    fare_std = np.mean([t[5] for t in episode_matrix])
    print "AVE_iter:%s\tEPISODE_REWARD:%s\tFINISHED_NUM:%s\tTIME_MEAN:%s\tTIME_STD:%s\tFARE_MEAN:%s\tFARE_STD:%s" \
        % (i, reward, finish_num, time_mean, time_std, fare_mean, fare_std)





