# How to use this folder for testing
The following instructions allow you to emulate the behaviour on the cluster

1. Set path variables correctly and make your working directory to be `cluster_setup/Scratch`
```bash
export TMPDIR="TMPDIR"
export HOME=".."
```

2. Make sure that rthe python interpreter being used is capable of running `cluster_setup/job_files/run.py`. Downloading `argminer==0.0.6` should work

3. Run the job script of your liking
```bash
/bin/bash job_script.sh
```