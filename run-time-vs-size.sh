#!/bin/bash

floyd run --gpu --env tensorflow-1.5 "python train.py --output-dir /output --task time_vs_size"