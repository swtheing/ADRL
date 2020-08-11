import sys
import os

if __name__ == "__main__":
    os.system("source /home/ssd5/DST/demo/test_env/bin/activate")
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

    gpuid = -1
    for param in param_list:
        gpuid += 1
        log_name = "".join(param) + "_exp06"
        command = "sh mc_run.sh %s %s" % (gpuid, log_name)
        for pa in param:
            command += " %s" % pa
        print command
        os.system(command)
    

