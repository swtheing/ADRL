cat $1 | grep -E "eval reward:|, Q :" > tmp_log
cat tmp_log | python tools.py $2 > $1.re_$2
#rm tmp_log
