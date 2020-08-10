import sys
for line in sys.stdin:
    group = line.strip().split(" ")
    print group[1] + "\t" + group[4] 
