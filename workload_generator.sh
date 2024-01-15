#!/bin/bash



#esempio comando
# ./workload_generator.sh -c config.json -e out -s mubench -r 180 -w 120 -n 3
# ./workload_generator.sh -d configs_test -e nuovi_esperimenti -s sockshop -r 10 -w 0 -n 5

#c: configurazione singola   (TODO: controllare mutua esclusione tra c e d)
#d: cartella di configurazioni
#e: cartella per l'esperimento, viene creata se non esiste
#s: system tra TrainTicket, muBench, SockShop (TODO: permettere di specificare un proprio init.sh)
#TODO: controllare presenza di flag obbligatori
#TODO: sceivere un help
#TODO: aggiungere alternative con long option
#TODO: controllare matrice loads muBench e poi cancellare i file
#TODO: controllare tutto il testo se è corretto in inglese

single_config=0

while getopts c:d:e:s:r:n:w: flag; do #ricorda che ':' significano che quella opzione richiede un argomento
	case "${flag}" in
	c) config_file=${OPTARG} single_config=1 ;;
	d) config_dir=${OPTARG} ;;
	e) out_dir=${OPTARG} ;;
	s) system=${OPTARG} ;;
	r) run_time=${OPTARG} ;;
	w) wait_time=${OPTARG} ;;
	n) num_rep=${OPTARG} ;;
	esac
done


cd $system
. ./init.sh #import host
cd ..


if [ $single_config -eq 1 ]; then
	configs=($config_file)
else
	configs=$(ls $config_dir | grep -E '*.json' | sed 's/.*/'$config_dir'\/&/')
fi

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
