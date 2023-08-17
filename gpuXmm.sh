#!/bin/bash

############################################################
# USAGE                                                    #
############################################################
echo_help_line()
{
  echo "    -h,--help             : print help about the command"
}
echo_all_line()
{
  echo "    -a,-all               : check all kernel"
}
echo_verbose_line()
{
  echo "    -v,--verbose          : enable verbose mod"
}
echo_mnp_line()
{
  echo "    -mXX, -nXX, -pXX      : dim m,n and p of the matrix multiplication"
  echo "                            if there is only one dimension set (e.g -m),"
  echo "                            the matrix multiplication will be"
  echo "                            c[mxm] = a[mxm] * b[mxm]"
  echo "                            if there are two dimensions set (e.g -m -p)," 
  echo "                            the matrix multiplication will be"
  echo "                            c[mxm] = a[mxp] * b[pxm]"
  echo "                            if there are tree dimensions set (e.g -m -p -n)," 
  echo "                            the matrix multiplication will be"
  echo "                            c[mxp] = a[mxn] * b[nxp]"
  echo "                            (default value = 100)"
}
echo_plot_line()
{
  echo "    -g,--plot={plot_file} : create a png plot with the results in the png file in argument"
  echo "                            (default: plot_file=./output/graphs/graph_DATE.png)"
}
echo_save_line()
{
  echo "    -s,--save={save_file} : save the measure output in the file in argument"
  echo "                            (default: save_file=./output/measure/measure_DATE.png)"
}
echo_prof_line()
{
  echo "    -P,--profiler         : profile the run using 'rocprof' or 'nsys prof'"
  echo "                            Output in output/profiler/"
}
echo_rdtsc_line()
{
  echo "    -r,--rdtsc           : print perfomance in RDTSC-Cycles instead of GFLOPS/s"
}
echo_force_line()
{
  echo "    -f,--force            : do not ask for starting a run"
}
echo_DPSP_line()
{
  echo "    -S,--SP               : run simple precision matrix mul (cumultative with -D)"
  echo "    -D,--DP               : run double precision matrix mul (cumultative with -S)"
}

echo_option_lines()
{
  for opt in $(echo $opt_list_short | sed -e 's/\(.\)/\1\n/g')
  do
    case $opt in
          h) echo_help_line ;;
          a) echo_all_line ;;
          f) echo_force_line ;;
          v) echo_verbose_line ;;
          P) echo_prof_line ;;
          r) echo_rdtsc_line ;;
          m) echo_mnp_line ;;
          s) echo_save_line ;;
          g) echo_plot_line ;;
          D) echo_DPSP_line ;;
      esac
  done
}

usage()
{  
  if [ "$cmd" == "check" ]; then
    usage_cmd "$cmd {options} -mXX -nXX -pXX <kernel>"
  elif [ "$cmd" == "measure" ]; then
    usage_cmd "$cmd {options} -mXX -nXX -pXX <kernel>"
  elif [ "$cmd" == "rank-update" ]; then
    usage_cmd "$cmd {options} <kernel>"
  fi  
  echo
  echo "Usage: $(basename $0) [cmd] {options} [args]"
  echo
  echo "cmd:"
  echo "    check       : check matrix multiply output"
  echo "    measure     : run normal benchmarks"
  echo "    rank-update : run a rank update benchmark"
  echo
  echo "options:"
  echo_verbose_line
  echo_help_line
  echo "kernels:"
  echo "    kernels are different depending the command (see $(basename $0) [cmd] -h)"
  echo
  exit 1
}

usage_cmd()
{
  echo
  echo "Usage: $(basename $0) $1"
  echo
  echo "options:"
  echo_option_lines
  echo "kernels:"
  echo "    ${kernel_list[*]}"
  echo
  exit 1
}

############################################################
# IF ERROR                                                 #
############################################################

check_error()
{
  err=$?
  if [ $err -ne 0 ]; then
    echo -e "gpuXmm: error in $0\n\t$1 ($err)"
    echo "Script exit."
    exit 1
  fi
}


############################################################
# YN                                                       #
############################################################
yn()
{
    while true; do
        read -p "$1 (y/n) " yn
      case $yn in
          [Yy]*) eval $2 ; break ;;
          [Nn]*) eval $3 ; exit 1 ;;
      esac
    done
}

############################################################
# ECHO VERBOSE                                             #
############################################################

eval_verbose()
{
  if [ $verbose == 1 ]; then
    eval $@
  fi
  if [ $verbose == 0 ]; then
    eval $@ > /dev/null
  fi
}

############################################################
# GET AVAILABLE KERNEL                                     #
############################################################
get_kernel_avail()
{
  kernel_list=( $(cat kernels_avail.config ) )
}


############################################################
# CHECK GPU                                                #
############################################################
check_gpu()
{
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
}

############################################################
# RUN COMMAND                                              #
############################################################

run_command()
{
    cmd=$1
    shift
    case $cmd in
        "check") 
              opt_list_short="havDSPm:n:p:" ; 
              opt_list="help,all,verbose,DP,SP,profiler" ; 
              run_check $@ ;;
        "measure" ) 
              opt_list_short="hfavPDSrm:n:p:s::g::" ; 
              opt_list="help,all,verbose,DP,SP,profiler" ; 
              run_measure $@ ;;
        "rank-update") 
              opt_list_short="hafSDvr" ; 
              opt_list="help,force,all,verbose,profiler,rdtsc,save::,plot:: " ; 
              run_rank_update $@ ;; 
        -h|--help) usage ;; 
        "") echo "gpuXmm: need command"; usage ;;
        *) echo "gpuXmm: $cmd is not an available command"; usage ;;
    esac
}

############################################################
# UTILS                                                    #
############################################################

get_matrix_dim()
{
  # Get the matrix dimension
  if [ ${#matrix_dim[@]} == 0 ]; then
    matrix_dim+=(100)
    matrix_dim+=(100)
    matrix_dim+=(100)
  fi
  if [ ${#matrix_dim[@]} == 1 ]; then
    matrix_dim+=(${matrix_dim[0]})
  fi
  if [ ${#matrix_dim[@]} == 2 ]; then
    matrix_dim+=(${matrix_dim[0]})
  fi
  m=${matrix_dim[0]}
  n=${matrix_dim[1]}
  p=${matrix_dim[2]}
  matrix_multiply_label="c["$m"x$p] = a["$m"x$n] * b["$n"x$p]"
}

get_kernels_to_run()
{
    # Get the kernels to run
    for i in $@; do 
      if [[ " ${kernel_list[*]} " =~ " `echo "$i" | tr '[:upper:]' '[:lower:]'` " ]] && [ $all == 0 ]; then
        kernel_to_run+="$i "
      fi
    done
    if [[ ${kernel_to_run[@]} == "" ]]; then
      echo "No kernel to run."
      exit 1
    fi
}

build_driver()
{
  eval_verbose echo "Build $kernel_lowercase kernel . . . "
  eval_verbose make $1 -B GPU=$GPU KERNEL=$kernel_uppercase METRIC=$metric_format PRECISION=$precision
  check_error "make failed"
}

init_option_var()
{
  verbose=0
  force=0
  all=0
  profiler=0
  plot=0
  save=0
  metric_format="GFLOPS/s"
  
  if [ $cmd == "measure" ]; then
    plot_file="$WORKDIR/output/graphs/graph_$(date +%F-%T).png"
    save_file="$WORKDIR/output/measure/measure_$(date +%F-%T).out"
  elif [ $cmd == "check" ]; then
    kernel_list=("${kernel_list[@]:1}")
  fi
}

check_option()
{
  init_option_var

  TEMP=$(getopt -o $opt_list_short \
                -l $opt_list \
                -n $(basename $0) -- "$@")
  if [ $? != 0 ]; then usage ; fi
  eval set -- "$TEMP"
  if [ $? != 0 ]; then usage ; fi

  while true ; do
      case "$1" in
          -h|--help) usage ;;
          -f|--force) force=1 ; shift ;;
          -a|--all) kernel_to_run=${kernel_list[@]} ; shift ;;
          -v|--verbose) verbose=1 ; shift ;;
          -P|--profiler) profiler=1 ; shift ;;
          -r|--rdtsc) metric_format="RDTSC-Cycles" ; shift ;;
          -m|-n|-p) matrix_dim+=($2); shift 2 ;;
          -S|--SP) precision_list+=("SP") ; shift ;;
          -D|--DP) precision_list+=("DP") ; shift ;;
          -s|--save) 
              case "$2" in
                  "") save=1; shift 2 ;;
                  *)  save=1; save_file="$2" ; shift 2 ;;
              esac ;;
          -g|--plot) 
              case "$2" in
                  "") plot=1; shift 2 ;;
                  *)  plot=1; plot_file="$2" ; shift 2 ;;
              esac ;;
          --) shift ; break ;;
          *) echo "No option $1."; usage ;;
      esac
  done

  if [ ${#precision_list[@]} == 0 ]; then
  precision_list=("DP")
  fi
}

############################################################
# RANKUPDATE FUNCTION                                      #
############################################################

run_rank_update()
{
  check_option $@
  get_kernels_to_run $@

  summary_measure
  
  for precision in ${precision_list[@]}
  do
    for i in $kernel_to_run
    do

      kernel_lowercase=`echo "$i" | tr '[:upper:]' '[:lower:]'`
      kernel_uppercase=`echo "$i" | tr '[:lower:]' '[:upper:]'`

      setup_measure_tmp_file
      build_driver "measure"
      for n in {1..32}
      do
        for m in {500..3000..100}
        do
            kernel_name=`printf "%16s , %4s ," "$kernel_lowercase" "$precision"`
            echo -n "$kernel_name" >> $measure_tmp_file
            matrix_multiply_label="c["$m"x$m] = a["$m"x$n] * b["$n"x$m]"
            measure_kernel $m $n $m
        done
      done


      plot_file="$WORKDIR/output/graphs/graph_"$kernel_lowercase"_"$precision"_$(date +%F-%T).png"
      echo "Plot generation . . ."
      python3 ./python/plot_gen_rankupdate_.py $measure_tmp_file $plot_file
      echo "Plot created in file $plot_file"

      echo "---------------------"
      echo "Result Summary : "
      cat $measure_tmp_file
      echo "---------------------"

      save_file="$WORKDIR/output/measure/rank_update_"$kernel_lowercase"_"$precision"_$(date +%F-%T).out"
      mv $measure_tmp_file $save_file
    done
    eval make clean --quiet
  done
}


############################################################
# CHECKER FUNCTION                                         #
############################################################

run_check()
{
  check_option $@
  get_kernels_to_run $@
  get_matrix_dim

  echo "Check matrix multiplication ($matrix_multiply_label) output. . ."
  for precision in ${precision_list[@]}
  do
    eval_verbose echo -e "Check base kernel . . ."
    eval_verbose make check KERNEL=BASIS GPU=$GPU PRECISION=$precision -B
    check_error "make echoué"
    eval ./check $m $n $p "./output/check/check_basis_$precision.out"
    check_error "run failed"

    for i in $kernel_to_run; do
      kernel_lowercase=`echo "$i" | tr '[:upper:]' '[:lower:]'`
      kernel_uppercase=`echo "$i" | tr '[:lower:]' '[:upper:]'`
      build_driver "check"
      check_kernel
    done

    make clean --quiet
  done
  echo "Vérifications terminées"
}

check_kernel()
{
  output_file="$WORKDIR/output/check/check_"$kernel_lowercase"_"$precision".out"
  cmd="$WORKDIR/check $m $n $p $output_file"
  eval_verbose echo "exec command : $cmd"

  if [ $profiler == 1 ]; then
    profiler_file="$kernel_lowercase_$m_$n_$p_$precision"
    profiler_dir=profiler_$profiler_file_$(date +%F-%T)
    mkdir $WORKDIR/output/profiler/$profiler_dir
    case "$GPU" in
        "NVIDIA") eval "nsys profile -o $WORKDIR/output/profiler/$profiler_dir/$profiler_file-rep $cmd" ;;
        "AMD") eval "rocprof --hip-trace --hsa-trace -o $WORKDIR/output/profiler/$profiler_dir/$profiler_file.csv $cmd" ;;
    esac
  else
    eval $cmd
  fi

  check_error "run check failed"
  echo "Check kernel $kernel_lowercase ($precision) : $(python3 $WORKDIR/python/check.py $WORKDIR/output/check/check_basis_$precision.out $output_file)"
  check_error "script python failed"
}


############################################################
# MEASURE FUNCTION                                         #
############################################################

run_measure()
{
  check_option $@
  get_kernels_to_run $@
  get_matrix_dim

  summary_measure
  
  setup_measure_tmp_file

  echo "Benchmark in progress . . ."
  for precision in ${precision_list[@]}
  do
    for i in $kernel_to_run; do
      kernel_lowercase=`echo "$i" | tr '[:upper:]' '[:lower:]'`
      kernel_uppercase=`echo "$i" | tr '[:lower:]' '[:upper:]'`
      kernel_name=`printf "%16s , %4s ," "$i" "$precision"`
      echo -n "$kernel_name" >> $measure_tmp_file
      build_driver "measure"
      measure_kernel $m $n $p
    done
  done
  echo "Benchmark finished"

  make clean --quiet

  if [ $plot == 1 ]; then
    echo "Plot generation . . ."
    python3 ./python/plot_gen_measure.py $metric_format $measure_tmp_file "$plot_file"
    echo "Plot created in file $plot_file"
  fi

  echo "---------------------"
  echo "Result Summary : "
  cat $measure_tmp_file | tr ',' '|'
  echo "---------------------"

  if [ $save == 1 ]; then
    mv $measure_tmp_file $save_file
  fi
}

setup_measure_tmp_file()
{
  measure_tmp_file=$WORKDIR/output/tmp/measure_tmp.out
  if [[ -f $measure_tmp_file ]]; then
  rm $measure_tmp_file
  fi
  if [ $metric_format == "RDTSC-Cycles"  ]; then
    echo "          kernel ,  prs ,     m ,     n ,     p ,        minimum ,         median ,      median/it ,   stab (%)" > $measure_tmp_file
  else
    echo "          kernel ,  prs ,     m ,     n ,     p ,        GLOPS/s ,   minimum (ms) ,     median (ms),   stab (%)" > $measure_tmp_file
  fi
}

summary_measure()
{
  echo -n "Summary measure : $matrix_multiply_label matrix multipy (${precision_list[@]}) on $GPU GPU"
  if [ $verbose == 1 ]; then
    echo -n " (with verbose mode)"
  fi 
  echo -e "\nKernel to measure : $kernel_to_run"
  if [ $metric_format == "RDTSC-Cycles" ]; then
    echo "Metrics : RDTSC metrics"
  else
    echo "Metrics : GFLOPS/s & ms metrics"
  fi
  if [ $plot == 1 ]; then
    echo "Plot will be generated in '$plot_file'"
  fi 
  if [ $save == 1 ]; then
    echo "Measure will be saved in '$save_file'"
  fi 
  if [ $force == 0 ]; then
    yn "Starting ?" "" "echo Aborted"
  fi
}

calculate_repetitions() 
{
    local matrix_size=$(( $1 * $2 ))

    if [ $matrix_size -gt 1000000 ]; then
      echo 50
    elif [ $matrix_size -lt 100 ]; then
      echo 1000
    else    
      local x_min=100
    local x_max=1000000
    local y_min=500
    local y_max=50

    echo $(( ($y_max - $y_min) * ($matrix_size - $x_min) / ($x_max - $x_min) + $y_min ))
    fi
}
measure_kernel()
{
  echo -e "Benchmark kernel $kernel_lowercase ($precision) ($matrix_multiply_label) . . ."
  is_stab_ok=false
  warmup=1000
  rep=100

  while [[ $is_stab_ok == false ]] ; do
    cmd="$WORKDIR/measure $1 $2 $3 $warmup $rep"
    eval_verbose echo "exec command : $cmd"

    if [ $profiler == 1 ]; then
      profiler_file="$kernel_lowercase_$1_$2_$3"
      profiler_dir=profiler_$profiler_file_$(date +%F-%T)
      mkdir $WORKDIR/output/profiler/$profiler_dir
      case "$GPU" in
          "NVIDIA") eval "nsys profile -o $WORKDIR/output/profiler/$profiler_dir/$profiler_file-rep $cmd" ;;
          "AMD") eval "rocprof --hip-trace --hsa-trace -o $WORKDIR/output/profiler/$profiler_dir/$profiler_file.csv $cmd" ;;
      esac
    else
      eval $cmd
    fi
    check_error "run measure failed"

    stab=$(tail -n 1 $measure_tmp_file | cut -d ',' -f 9 | sed 's/ //g')
    if (( $(echo "$stab < 5" | bc -l) ||  $rep > 1000 )); then
      is_stab_ok=true
    else
      rep=$(( $rep*10 ))
      echo "Bad stability : Re-measure with $rep repetitions :"
      sed -i '$ d' $measure_tmp_file
      echo -n "$kernel_name" >> $measure_tmp_file
    fi
  done
}

############################################################
# MAIN                                                     #
############################################################

# check_gpu
WORKDIR=`realpath $(dirname $0)`
cd $WORKDIR
get_kernel_avail

run_command $@

