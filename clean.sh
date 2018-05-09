rm -vrf ./__pycache__/
rm -vrf ./utils/__pycache__/
rm -vrf ./models/__pycache__/

log_list=`find . -maxdepth 1 -name 'log_*'`
if [ -n "$log_list" ]; then
  echo -e "\033[1m\033[95m"
  echo "Log Exist, Move or Remove It!"
  echo "$log_list"
  echo -e "\033[0m\033[0m"
fi

dump_list=`find . -maxdepth 1 -name 'dump_*'`
if [ -n "$dump_list" ]; then
  echo -e "\033[1m\033[95m"
  echo "Dump Exist, Move or Remove It!"
  echo "$dump_list"
  echo -e "\033[0m\033[0m"
fi

visual_list=`find . -maxdepth 1 -name 'visual_*'`
if [ -n "$visual_list" ]; then
  echo -e "\033[1m\033[95m"
  echo "Visual Exist, Move or Remove It!"
  echo "$visual_list"
  echo -e "\033[0m\033[0m"
fi