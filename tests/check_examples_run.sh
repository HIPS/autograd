#!/bin/bash

PYTHONPATH=".:$PYTHONPATH"
trap 'kill -INT -$pid && exit 1' INT

working=()
failing=()

examples=$(find examples -name '*.py' -not -name '__init__.py')

echo 'Running all the examples...'
for f in $examples; do
    timeout 15s python2 $f > /dev/null 2>&1 & pid=$!
    wait $pid
    status=$?
    if [ $status -eq 0 -o $status -eq 124 ]; then
        echo $f "seems to work"
        working+=($f)
    elif [ $status -eq 137 ]; then
        echo $f "might be working, but had to be killed"
        working+=($f)
    else
        echo $f "seems broken, try running manually"
        failing+=($f)
    fi
done

if [ ! ${#working[@]} -eq 0 ]; then
    echo -e '\033[01;36m'
    echo "These seemed to WORK:"
    echo -en '\033[00m'
    printf '%s\n' "${working[@]}"
    echo
fi
if [ ! ${#failing[@]} -eq 0 ]; then
    echo -e '\033[01;31m'
    echo "These seemed to FAIL:"
    echo -en '\033[00m'
    printf '%s\n' "${failing[@]}"
    echo
fi
