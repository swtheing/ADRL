import sys
count = 0
epoch = 1
sum = 0.0 
for line in sys.stdin:
    if "eval" not in line:
        #print line.strip()
        continue
    groups = line.strip().split()
    if "sss" in line:
        continue
    sum += float(groups[-1])
    count += 1
    if count == float(sys.argv[1]):
        print ("iter %d eval reward: %f"% (epoch, sum/float(sys.argv[1])))
        epoch += 1
        count = 0
        sum = 0.0
