cat $1 | grep -E "eval reward:" > tmp_log
#cat $1 | grep -E "INNER SETP:10, last_reward:" > tmp_log
cat tmp_log | python tools.py $2 > $1.re_$2
rm tmp_log
