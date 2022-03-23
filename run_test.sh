#!/bin/bash

command="nice python3 distributer.py arxiv BigBird 128 0.10 results"
echo $command
eval $command
