#!/usr/bin/env bash

rm -rf ./build
rm -rf ./dist
rm -rf ./magnetron.egg-info
pip3 wheel --verbose -w dist .
