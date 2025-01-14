#!/usr/bin/env bash

# Run this script before running the benchmark
sudo sysctl kernel.perf_event_paranoid=1
sudo cpufreq-set -g performance
