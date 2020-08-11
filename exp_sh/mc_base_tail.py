import sys
import os

if __name__ == "__main__":
    root_path = "./mc_log_exp04/"
    #os.system("source /home/ssd5/DST/demo/test_env/bin/activate")
    exp_conf = open("mc_exp_conf.txt", "r")
    param_list = []
    for line in exp_conf:
        #print line.strip()
        if "#" in line:
            continue
        terms = line.strip().split("\t")
        if len(terms) >= 3:
            # max_step = int(terms[0])
            # task_num = int(terms[1])
            # par_num = int(terms[2])
            param = terms[0:3]
            param_list.append(param)
    exp_conf.close()

    result = []
    for param in param_list:
        log_name = "".join(param) #+ "_exp06"
        real_log_name = log_name + "_mc"
        if not os.path.exists(root_path + real_log_name):
            log_name = "".join(param) + "_exp04"
            real_log_name = log_name + "_mc"
            if not os.path.exists(root_path + real_log_name):
                continue
        log_f = open(root_path + real_log_name, "r")
        max_reward = 0.0
        result_str = ""
        head = "\t".join(param)
        for line in log_f:
            line = line.strip()
            if "iter" not in line:
                continue
            if "eval reward:" in line:
                terms = line.split(", ")
                
                ts = terms[0].split(": ")
                reward = float(ts[1])

                ts = terms[1].split(": ")
                raw_reward = float(ts[1])

                ts = terms[2].split(": ")
                f_std_reward = float(ts[1])

                ts = terms[3].split(": ")
                t_std_reward = float(ts[1])

                ts = terms[4].split(": ")
                dis_cost = float(ts[1])

                if reward > max_reward:
                    max_reward = reward
                    result_str = "%s\t%s\t%s\t%s\t%s\t%s" % (head, reward, raw_reward, f_std_reward, t_std_reward, dis_cost)
        result.append(result_str)
        
    for res in result:
        print res


