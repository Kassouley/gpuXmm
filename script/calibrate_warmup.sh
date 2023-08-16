#!/bin/bash

############################################################
# USAGE                                                    #
############################################################
usage()
{
  usage="
  \nUsage: $0 {options} <problem size> <kernels>\n\n
  options:\n
    \t-h,--help : print help\n
    \t-a,--all : calibrate all kernel\n
    \t-n,--nbstep=XX : number of step for the calibration (default value = $nb_step)\n
    \t-p,--plot: create a png plot with the results in png file in  ./graphs/calibrate_warmup_SIZE_KERNEL_DATE.png)\n
    \t-v,--verbose : print all make output\n
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
build_kernel()
{
  kernel=`echo "$1" | tr '[:lower:]' '[:upper:]'`
  echo -e -n "Build kernel $1 . . . "
  eval make calibrate -B KERNEL=$kernel CLOCK=RDTSC GPU=$GPU $output
  check_error "build failed"
  echo "Done"
}

calibrate_kernel()
{
  for n in ${data_size[@]}; do
    if [[ $n =~ $re ]]; then
      echo -e "Calibrate kernel $1 (problem size: $n, nb step: $2) . . ."
      cmd="$WORKDIR/calibrate $n $2 $output_file"
      eval echo "exec command : $cmd" $output
      eval $cmd
      check_error "run calibrate failed"
      generate_plot $n $1
    fi
  done
}

generate_plot()
{
  if [ $plot == 1 ]; then
    plot_file="$WORKDIR/graphs/warmup/warmup_"$2"_"$1"_$(date +%F-%T).png"
    echo "Graph generation . . ."
    python3 ./python/graph-gen-warmup.py $1 $output_file $plot_file $clock_label
    echo "Graph created in file $plot_file"
  fi
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
kernel_to_calibrate=()
data_size=()
nb_step=20
verbose=0
output="> /dev/null"
force=0
all=0
plot=0
clock="RDTSC"
clock_label="RDTSC-cycles"

TEMP=$(getopt -o hfavpn: \
              -l help,force,all,verbose,plot,nbstep: \
              -n $(basename $0) -- "$@")

eval set -- "$TEMP"

if [ $? != 0 ]; then usage ; fi

while true ; do
    case "$1" in
        -a|--all) kernel_to_calibrate=(${kernel_list[@]}) ; shift ;;
        -f|--force) force=1 ; shift ;;
        -v|--verbose) verbose=1 ; shift ;;
        -p|--plot) plot=1; shift ;;
        -n|--nbstep) nb_step=$2 ; shift 2 ;;
        --) shift ; break ;;
        -h|--help) usage ;;
        *) echo "No option $1."; usage ;;
    esac
done

############################################################
# CHECK ARGS                                               #
############################################################
re='^[0-9]+$'
for i in $@; do 
  if [[ " ${kernel_list[*]} " =~ " ${i} " ]] && [ $all == 0 ]; then
    kernel_to_calibrate+=("$i")
  elif [[ $i =~ $re ]]; then
    data_size+=("$i")
  fi
done
if [[ ${data_size[@]} == "" ]]; then
  data_size=100
fi

if [[ ${kernel_to_calibrate[@]} == "" ]]; then
  echo "No kernel to calibrate."
  exit 1
fi
kernel_to_calibrate=`echo "$kernel_to_calibrate" | tr '[:upper:]' '[:lower:]'`

############################################################
# SUMMARY OF THE RUN                                       #
############################################################
echo -n "Summary calibrate on $GPU GPU"
if [ $verbose == 1 ]; then
  output=""
  echo -n " (with verbose mode)"
fi 
echo -e "\nKernel to calibrate (${#kernel_to_calibrate[*]}) : ${kernel_to_calibrate[*]}"
echo -e "Problem size : ${data_size[*]}"
echo -e "Step number: $nb_step"
if [ $plot == 1 ]; then
  echo "Plot will be generated"
else
  echo "Plot will NOT be generated"
fi 
duration=$((${#kernel_to_calibrate[@]} * 3 * ${#data_size[@]}));
echo "Approximately $duration minutes long"

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
# START CALIBRATION                                        #
############################################################
output_file="$WORKDIR/output/tmp/calibration_warmup_tmp.out"
if [[ -f $output_file ]]; then
  rm $output_file
fi
touch $output_file

for i in ${kernel_to_calibrate[*]}; do
  if [[ " ${kernel_list[*]} " =~ " ${i} " ]]; then
    build_kernel $i
    calibrate_kernel $i $nb_step
  fi
done

echo "Calibration finished"

eval make clean $output




