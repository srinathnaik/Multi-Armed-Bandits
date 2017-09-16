#!/bin/bash

# $1->horizon, $2->randomseed

horizon=( 10 100 1000 10000 100000 )

runs=100
seed=0

cmd="./startexperiment.sh $horizon $seed"

for i in "${horizon[@]}"
do

	total_regret=0.0
	regret=0.0
	for count in `seq 1 $runs`;
	do
		seed=$RANDOM
		cmd="./startexperiment.sh $i $seed"
		$cmd > /dev/null
		regret=`tail -2 serverlog.txt | head -1 | awk '{ print $3 }'`
		# let total_regret=total_regret+regret
		total_regret=$(bc <<< "$total_regret+$regret")

		#echo "H" $horizon "R" $count "S" $seed "Regret" $regret 
	done

	avg=`echo "scale=4 ; $total_regret / $runs" | bc`
	echo "H" $i "R" $count "Average_Regret" $avg "Tot_regret" $total_regret

done