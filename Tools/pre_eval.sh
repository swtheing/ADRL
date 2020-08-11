cat $1 | grep -E "eq_match" > tmp_log
cat tmp_log | python tools_pre.py $2 > $1.re_$2
#rm tmp_log
