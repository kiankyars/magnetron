@echo off
if exist build (
    rmdir /s /q build
)
if exist dist (
    rmdir /s /q dist
)
if exist magnetron.egg-info (
    rmdir /s /q magnetron.egg-info
)
pip3 wheel --verbose -w dist .
