#!/bin/bash

PWD=`pwd`

horizon=$1
port=5001
nRuns=100
hostname="localhost"
banditFile="$PWD/data/instance-25.txt"

algorithm="epsilon-greedy"
# Allowed values for algorithm parameter(case-sensitive)
# 1. epsilon-greedy 
# 2. UCB 
# 3. KL-UCB 
# 4. Thompson-Sampling
# 5. rr

epsilon=0.025

numArms=$(wc -l $banditFile | cut -d" " -f1 | xargs)

SERVERDIR=./server
CLIENTDIR=./client

OUTPUTFILE=$PWD/serverlog.txt

randomSeed=$2

pushd $SERVERDIR
cmd="./startserver.sh $numArms $horizon $port $banditFile $randomSeed $OUTPUTFILE &"
#echo $cmd
$cmd 
popd

sleep 1

pushd $CLIENTDIR
cmd="./startclient.sh $numArms $horizon $hostname $port $randomSeed $algorithm $epsilon&"
#echo $cmd
$cmd > /dev/null 
popd

