set USE_CYTHON=True

"%PYTHON%" -c "from version import __version__; print(__version__)" > __conda_version__.txt
if %ERRORLEVEL% NEQ 0 exit /B 1

"%PYTHON%" setup.py install
if %ERRORLEVEL% NEQ 0 exit /B 1
