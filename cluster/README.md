# Cluster compute testing directory

This is a directory that lightly mocks the structure in a cluster computer system. This allows you to test .sh scripts for syntax errors and correct behaviour before submitting jobs.

Note that depending on the cluster compute system you are using that this structure may vary.

## How to use this folder for testing

Please note that the directory structure as to be as follows:

- cluster_setup/
    - Scratch/
        - data/
            - train-test-split.csv (must exist if using TUDarmstadt)
            - {labelling_strategy}/
                - {dataset}_postprocessed.json
        - job_files/
            - run.py
        - TMPDIR/
        - job_script.sh

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