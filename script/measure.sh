#!/bin/bash

############################################################
# USAGE                                                    #
############################################################
usage()
{
  usage="
  \nUsage: $0 {options} [problem size] <kernels>\n\n
  options:\n
    \t-h,--help : print help\n
    \t-a,-all : measure all kernel\n
    \t-p,--plot={plot_file} : create a png plot with the results in png file in argument (default: ./graphs/graph_DATE.png)\n
    \t-s,--save={out_file} : save the measure output in file in argument (default: ./output/measure_DATE.out)\n
    \t-v,--verbose : print all make output\n
    \t-P,--profiler : profile the run using 'rocprof' or 'nsys prof'. Output in output/profiler/\n
    \t-m,--milliseconds : time is in milliseconds instead of RTDSC-Cycles\n
    \t-f,--force : do not ask for starting the measure\n
  problem size :\n\tdefault value = 100\n
  kernels:\n
    \t${kernel_list[*]}\n"

  echo -e $usage
  exit 1
}

############################################################
# IF ERROR                                                 #
############################################################

check_error()
{
  err=$?
  if [ $err -ne 0 ]; then
    echo -e "gpumm: error in $0\n\t$1 ($err)"
    echo "Script exit."
    exit 1
  fi
}

############################################################
# CHECK REQUIERMENT                                        #
############################################################
check_requierment()
{
  JQ_OK=$(which jq)
  if [ "" = "$JQ_OK" ]; then
    echo -n "jq is needed. Install jq ?"
    while true; do
          read -p " (y/n) " yn
          case $yn in
              [Yy]*) sudo apt-get --yes install jq ; break;; 
              [Nn]*) echo "Aborted" ; exit 1 ;;
          esac
      done
  fi
}

############################################################
# FUNCTION                                                 #
############################################################


measure_kernel()
{
  if [[ $2 -ne 10 && $2 -ne 100 && $2 -ne 1000 ]]; then
    data_size="default"
  else
    data_size=$2
  fi
  build_kernel $1
  echo -e "Measure kernel $1 (problem size: $2) . . ."
  config="$WORKDIR/json/measure_config.json"
  key=".$GPU.$1[\"$data_size\"]"
  warmup=$( jq $key.warmup $config )
  rep=$( jq $key.rep $config )
  cmd="$WORKDIR/measure $2 $warmup $rep"
  eval echo "exec command : $cmd" $output
  if [ $profiler == 1 ]; then
    profiler_dir=profiler_$1_$2_$(date +%F-%T)
    mkdir $WORKDIR/output/profiler/$profiler_dir
    case "$GPU" in
        "NVIDIA") eval "nsys profile -o $WORKDIR/output/profiler/$profiler_dir/$1_$2-rep $cmd" ;;
        "AMD") eval "rocprof --hip-trace --hsa-trace -o $WORKDIR/output/profiler/$profiler_dir/$1_$2.csv $cmd" ;;
    esac
  else
    eval $cmd
  fi
  check_error "run measure failed"
}

build_kernel()
{
  kernel=`echo "$1" | tr '[:lower:]' '[:upper:]'`
  echo -e -n "Build kernel $1 . . . "
  eval make measure -B KERNEL=$kernel CLOCK=$clock GPU=$GPU $output
  check_error "build failed"
  echo "Done"
}

############################################################
# SETUP                                                    #
############################################################

check_requierment
WORKDIR=`realpath $(dirname $0)/..`
cd $WORKDIR

############################################################
# CHECK GPU                                                #
############################################################
GPU_CHECK=$( lshw -C display 2> /dev/null | grep nvidia )
GPU=NVIDIA
if [[ -z "$GPU_CHECK" ]]; then
  GPU_CHECK=$( lshw -C display 2> /dev/null | grep amd )
  GPU=AMD
fi
if [[ -z "$GPU_CHECK" ]]; then
  echo "No GPU found."
  exit 1
fi

############################################################
# CHECK OPTIONS                                            #
############################################################

kernel_list=( $(jq ".$GPU|keys_unsorted[]" json/measure_config.json -r) )
kernel_to_measure=()
verbose=0
output="> /dev/null"
force=0
all=0
clock="RDTSC"
clock_label="RDTSC-cycles"
plot=0
plot_file="$WORKDIR/output/graphs/graph_$(date +%F-%T).png"
save=0
save_file="$WORKDIR/output/measure/measure_$(date +%F-%T).out"
profiler=0

TEMP=$(getopt -o hfavPms::p:: \
              -l help,force,all,verbose,profiler,millisecond,save::,plot:: \
              -n $(basename $0) -- "$@")

eval set -- "$TEMP"

if [ $? != 0 ]; then usage ; fi

while true ; do
    case "$1" in
        -a|--all) kernel_to_measure=${kernel_list[@]} ; shift ;;
        -f|--force) force=1 ; shift ;;
        -v|--verbose) verbose=1 ; shift ;;
        -P|--profiler) profiler=1 ; shift ;;
        -m|--millisecond) clock="MS" ; clock_label="Time (ms)"; shift ;;
        -s|--save) 
            case "$2" in
                "") save=1; shift 2 ;;
                *)  save=1; save_file="$2" ; shift 2 ;;
            esac ;;
        -p|--plot) 
            case "$2" in
                "") plot=1; shift 2 ;;
                *)  plot=1; plot_file="$2" ; shift 2 ;;
            esac ;;
        --) shift ; break ;;
        -h|--help) usage ;;
        *) echo "No option $1."; usage ;;
    esac
done

############################################################
# CHECK ARGS                                               #
############################################################
re='^[0-9]+$'
data_size=100
for i in $@; do 
  if [[ " ${kernel_list[*]} " =~ " ${i} " ]] && [ $all == 0 ]; then
    kernel_to_measure+=" $i"
  elif [[ $i =~ $re ]]; then
    data_size="$i"
  fi
done

if [[ ${kernel_to_measure[@]} == "" ]]; then
  echo "No kernel to measure."
  exit 1
fi
kernel_to_measure=`echo "$kernel_to_measure" | tr '[:upper:]' '[:lower:]'`

############################################################
# SUMMARY OF THE RUN                                       #
############################################################
echo -n "Summary measure ($clock_label) on $data_size x $data_size matrix on $GPU GPU"
if [ $verbose == 1 ]; then
  output=""
  echo -n " (with verbose mode)"
fi 
echo -e "\nKernel to measure : $kernel_to_measure"
if [ $plot == 1 ]; then
  echo "Plot will be generated in '$plot_file'"
fi 
if [ $save == 1 ]; then
  echo "Measure will be saved in '$save_file'"
fi 

if [ $force == 0 ]; then
  echo -n "Starting ?"
  while true; do
          read -p " (y/n) " yn
        case $yn in
            [Yy]*) break ;;
            [Nn]*) echo "Aborted" ; exit 1 ;;
        esac
    done
fi

############################################################
# START MEASURE                                            #
############################################################
if [[ -f $WORKDIR/output/tmp/measure_tmp.out ]]; then
  rm $WORKDIR/output/tmp/measure_tmp.out
fi
echo "     kernel    |     minimum     |     median     |   median/it   |   stability (%)" > $WORKDIR/output/tmp/measure_tmp.out


echo "Measures in progress . . ."
for i in $kernel_to_measure; do
  if [[ " ${kernel_list[*]} " =~ " ${i} " ]]; then
    kernel_name=`printf "%14s" "$i"`
    echo -n "$kernel_name" >> $WORKDIR/output/tmp/measure_tmp.out
    measure_kernel $i $data_size
  fi
done
echo "Measures finished"

eval make clean $output

if [ $plot == 1 ]; then
  echo "Graph generation . . ."
  python3 ./python/graph-gen-measure.py $data_size $WORKDIR/output/tmp/measure_tmp.out "$plot_file" "$clock_label"
  echo "Graph created in file $plot_file"
fi

echo "---------------------"
echo "Result Summary ($clock_label): "
cat $WORKDIR/output/tmp/measure_tmp.out
echo "---------------------"

if [ $save == 1 ]; then
  mv $WORKDIR/output/tmp/measure_tmp.out $save_file
fi

