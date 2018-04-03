#!/bin/bash

floyd run --cpu --env tensorflow-1.5 "python train.py --output-dir /output --task trace"