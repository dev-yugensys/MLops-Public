@echo off
echo Installing test requirements...
pip install -r requirements-test.txt

echo Running tests...
python -m pytest tests/ -v --cov=pages --cov-report=term-missing

pause
