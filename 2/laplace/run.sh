#!/bin/bash
for i in 1 2 5 10 20 40;
  do echo -n "thread ${i}t - " && /opt/main $N $i;
done;
