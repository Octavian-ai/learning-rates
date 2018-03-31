#!/bin/bash

floyd run --gpu --env tensorflow-1.5 "python train.py --output-dir /output --max-mins 4"