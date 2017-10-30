#!/bin/bash

PYTHONPATH=".:$PYTHONPATH"

for f in examples/*.py; do
    timeout 15s python $f > /dev/null
    status=$?
    if [ $status -eq 0 -o $status -eq 124 ]; then
        echo $f "seems to work"
    elif [ $status -eq 137 ]; then
        echo $f "might be working, but had to be killed"
    else
        echo $f "seems broken, try running manually"
    fi
done
