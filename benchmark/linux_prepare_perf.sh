#!/usr/bin/env bash

# Run this script before running the benchmark

sudo -S sysctl kernel.perf_event_paranoid=1
sudo cpufreq-set -g performance
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
