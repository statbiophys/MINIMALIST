import subprocess
from sys import argv

model=str(argv[1])

subprocess.run('python scripts/simulate.py '+model, shell=True)
subprocess.run('python scripts/infer_hyperpars.py '+model, shell=True)
subprocess.run('python scripts/infer_estimators.py '+model, shell=True)
subprocess.run('python scripts/compare_estimators.py '+model, shell=True)