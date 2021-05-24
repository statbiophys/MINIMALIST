import subprocess
from sys import argv

model=str(argv[1])
n_sims=[17e6,2e6,2e5,2e4]
objectives=['MINE','BCE','fMINE']

for n in n_sims:
    to_run=[]   
    for obj in objectives:
        code=model+' '+str(int(n))+' '+obj
        to_run.append('python scripts/infer.py '+code)                                                       
    subprocess.run('& '.join(to_run), shell=True)