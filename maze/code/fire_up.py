#!/bin/bash -l
import os 

work_dir = '/home/gantonov/code/python/'
out_dir  = '/home/gantonov/data/'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# --- simulation parameters ---
env    = 'tolman'
nsteps = 500
beta   = 10

for seed in range(15):
    
    job_file = os.path.join(os.getcwd(), 'sim_%u.sh'%seed)

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash -l\n")
        fh.writelines("#SBATCH -J sim_%u\n"%seed)
        fh.writelines("#SBATCH -o " + os.path.join(out_dir, 'job.out.%j') + "\n")
        fh.writelines("#SBATCH -e " + os.path.join(out_dir, 'job.err.%j') + "\n")
        fh.writelines("#SBATCH -D " + work_dir + "\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --cpus-per-task=1\n")
        fh.writelines("#SBATCH --mem-per-cpu=2000\n")
        fh.writelines("#SBATCH --time=24:00:00\n")
        
        fh.writelines("module purge\n")
        fh.writelines("conda activate ai\n")
        fh.writelines("srun python main_replay.py -e %s -ns %u -b %u -s %u"%(env, nsteps, beta, seed))
    
    os.system("sbatch %s"%job_file)