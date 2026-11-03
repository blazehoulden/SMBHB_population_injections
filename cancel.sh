#!/bin/bash
FIRST=18868448   # realistic/pessimistic orchestrator IDs and above
W=./cancel_work
mkdir -p "$W"

squeue -u $USER -h -o "%F|%i|%j|%t|%E" > $W/q.txt

# stage1 arrays with a running task -> keep
awk -F'|' -v f=$FIRST '$1>f && $3 ~ /^s1_/ && $4=="R" {print $1}' $W/q.txt | sort -u > $W/keep_s1.txt

# everything pending that is not kept
awk -F'|' -v f=$FIRST '
  NR==FNR {keep[$1]=1; next}
  $1>f && $4=="PD" {
    if ($3 ~ /^s1_/) { if (!($1 in keep)) print $1 }
    else if ($3 ~ /^s2_/) {
      k=0; n=split($5,a,/[^0-9]+/)
      for(i=1;i<=n;i++) if (a[i] in keep) k=1
      if (!k) print $2
    }
  }' $W/keep_s1.txt $W/q.txt | sort -u > $W/cancel.txt