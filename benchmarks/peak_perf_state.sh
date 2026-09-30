cat /sys/class/power_supply/A*/online # verify AC power
rocm-smi -d 0 --setperflevel high 
rocm-smi -d 0 --showclocks # verify perf level is high