#!/bin/bash
echo -n "sequential 1t - " && /opt/pi_seq $N
for i in 1 2 5 10 20 40;
  do echo -n "thread ${i}t - " && /opt/pi_pth $N $i;
done;
for i in 1 2 5 10 20 40;
  do echo -n "omp ${i}t - " && /opt/pi_omp $N $i;
done;
