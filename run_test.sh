#!/bin/bash

command="nice python3 distributer.py arxiv LED 64 .05 results"
echo $command
eval $command
