#!/bin/bash



#esempio comando
# ./workload_generator.sh -c config.json -e out -s mubench -r 180 -w 120 -n 3
# ./workload_generator.sh -d configs_test -e nuovi_esperimenti -s sockshop -r 10 -w 0 -n 5

#c: configurazione singola
#d: cartella di configurazioni
#e: cartella per l'esperimento, viene creata se non esiste
#s: system tra trainticket, mubench, sockshop (TODO: permettere di specificare un proprio init.sh)

#TODO: sceivere un help
#TODO: aggiungere alternative con long option
#TODO: controllare tutto il testo se è corretto in inglese

while getopts c:d:e:s:r:n:w: flag; do #ricorda che ':' significano che quella opzione richiede un argomento
	case "${flag}" in
	c) config_file=${OPTARG} ;;
	d) config_dir=${OPTARG} ;;
	e) out_dir=${OPTARG} ;;
	s) system=${OPTARG} ;;
	r) run_time=${OPTARG} ;;
	w) wait_time=${OPTARG} ;;
	n) num_rep=${OPTARG} ;;
	*) echo 'Error in command line parsing' >&2; exit 1
	esac
done

if [ -z "$config_file" ] && [ -z "$config_dir" ]; then
        echo 'Missing -c or -d' >&2
        exit 1
fi

if [ ! -z "$config_file" ] && [ ! -z "$config_dir" ]; then
        echo 'Only one between -c and -d is accepted' >&2
        exit 1
fi

if [ -z "$out_dir" ]; then
        echo 'Missing -s' >&2
        exit 1
fi

if [ -z "$system" ]; then
        echo 'Missing -e' >&2
        exit 1
fi

if [ -z "$run_time" ]; then
        echo 'Missing -r' >&2
        exit 1
fi

if [ -z "$wait_time" ]; then
        echo 'Missing -w' >&2
        exit 1
fi

if [ -z "$num_rep" ]; then
        echo 'Missing -n' >&2
        exit 1
fi

if [ ! -z "$config_file" ]; then
    if [ ! -f "$config_file" ]; then
        echo "$config_file does not exists." >&2
        exit 1
    fi
fi

if [ ! -z "$config_dir" ]; then
    if [ ! -d "$config_dir" ]; then
        echo "$config_dir does not exists.">&2
        exit 1
    else
        if [ -z "$(find $config_dir -mindepth 1 -maxdepth 1)" ]; then
            echo "$config_dir is empty.">&2
            exit 1
        fi
    fi
fi

if [ -z "$config_dir" ]; then
	configs=($config_file)
else
	configs=$(ls $config_dir | grep -E '*.json' | sed 's/.*/'$config_dir'\/&/')
	if [ ${#configs[@]} -eq 0 ] || [ -z ${configs[0]} ]; then
	    echo "There are not configs in $config_dir">&2
	    exit
	fi
fi

cd $system
. ./init.sh #import host
cd ..

for cf in $configs; do
	filename=$(basename -- "$cf" .json)
	echo "Running "$filename

	readarray -t nusers < <(jq -c '.nusers[]' $cf)
	readarray -t loads < <(jq -c '.loads[]' $cf | sed 's/"//g')
	readarray -t SRs < <(jq -c '.spawn_rates[]' $cf)

	n_exp=$((${#nusers[@]} * ${#loads[@]} * ${#SRs[@]}))
	cur_exp=0

	cdir=$out_dir/$filename
	mkdir -p $cdir

	for sr in "${SRs[@]}"; do

		dir=$cdir"/experiments_sr_""$sr"
		mkdir -p "$dir"

		for u in "${nusers[@]}"; do
			u_dir=$dir/users_$u
			mkdir -p "$u_dir"

			for l in "${loads[@]}"; do
				cur_exp=$((cur_exp + 1))
				l_dir=$u_dir/$l
				mkdir -p $l_dir

                sleep $wait_time
				echo "Run Experiment("$cur_exp"/"$n_exp"):"$u"_"$l"_"$sr

				for ((r = 1; r <= "$num_rep"; r++)); do
					sleep $wait_time
					echo "Start Replica"
					locust -f $system"/loads/locustfile_"$l".py" --skip-log-setup --headless --users $u --spawn-rate $sr -H $host --run-time $run_time"s" --csv=$l_dir"/esec_"$r".csv" &

					BACK_PID=$!

					for ((t = 1; t <= $run_time; t++)); do
						kubectl top pod -n $system >>$l_dir/mem_cpu_$r.txt
						sleep 1
					done

					wait $BACK_PID #wait locust finish
					echo "FINE REPLICA"
				done
			done
		done
	done

done

cd $system
. ./stop_sys.sh
cd ..
