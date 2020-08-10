import sys
import os

if __name__ == "__main__":
    root_path = "./0104/"
    #os.system("source /home/ssd5/DST/demo/test_env/bin/activate")
    exp_conf = open("baseline_exp_conf.txt", "r")
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

    result_list = ["NA", "NA", "NA", "NA", "NA"]
    res_detail = ["NA", "NA", "NA", "NA"]
    log_dict = {
        "r_" : [0, 0, "rand"],
        "g_" : [2, 1, "greedy"],
        "go_" : [3, 2, "greedy_opt"],
        "w_" : [4, 3, "worst_off"]
    }
    for param in param_list:
        log_name = "".join(param) #+ "_exp"
        for appendix in log_dict:
            real_log_name = appendix + log_name
            final_r = 0.0
            max_r = 0.0
            log_f = open(root_path + real_log_name, "r")
            res_scores = []
            for line in log_f:
                line = line.strip()
                if "ave_reward:" in line:
                    terms = line.split(", ")
                    counter = 0
                    for t in terms:
                        counter += 1
                        score = t.split(":")[1]
                        res_scores.append(score)
                        if counter == 1:
                            final_r = score
                        if counter == 2:
                            max_r = score
                if "mean_raw_reward:" in line:
                    terms = line.split(", ")
                    counter = 0
                    for t in terms:
                        counter += 1
                        score = t.split(":")[1]
                        res_scores.append(score)
            if len(res_scores) != 8:
                print "RES error! %s" % real_log_name
            index = log_dict[appendix][0]
            index2 = log_dict[appendix][1]
            typ = log_dict[appendix][2]
            if typ == "rand":
                result_list[index + 1] = max_r
            result_list[index] = final_r
            res_detail[index2] = "\t".join(res_scores)

        result = "\t".join(param) + "\t" + "\t".join(result_list)
        res_detail_str = "\t".join(param)
        for res in res_detail:
            res_detail_str += "\t" + res

        print res_detail_str


