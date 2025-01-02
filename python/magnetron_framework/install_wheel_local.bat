@echo off
call build_wheel.bat
for %%f in (dist\*.whl) do (
    pip3 install "%%f" --force-reinstall
)