# Contributing Instructions

## Set-up

First, clone the repository locally. After entering the repository, create a new virtual environment and run:

`pip install -e .`

If you now try `pip list` you should see `ArgMiner` as one of your installed packages. You have just installed ArgMiner in a development environment. So any dependencies packaged with the previous version of ArgMiner have now been installed.

In order to now 'develop' in this environment, you can update your virtual env with any new packages using `pip install`. The key thing is that these won't be persisted into ArgMiner, but they will be used by you locally. This will help us keep the production version clean, while we change requirements ourselves.

I've added a `requirements.txt` file in every folder. Make sure that when you install something new, that you keep the folder updated e.g. using `pip freeze > your_name/requirements.txt`.

## Using Jupyter Notebooks

First, install `notebook` into your virtual environment using `pip install notebook`.

Now, unlike with Python, you won't be able to directly import anything from ArgMiner because Jupyter Notebooks limit the path to the directory that you are running the notebook from. This means that it won't be able to 'see' the ArgMiner package. In order to let it do this, you must add your virtual env as a kernel to Jupyter. First, run:

`pip install ipykernel`

Then:

`ipython kernel install --name==your_venv_name`

If you now go into Jupyter, when selecting the kernel with which to run your notebook make sure that it is the same as the one you just installed! You should now be able to import from ArgMiner as a package.

## Using Colab

Unfortunately because of the way Colab works, using it with our custom environment is a bit difficult. In other projects, the way we did it was by zipping the entire project each time it was updated, then unzipping it on Colab, and then installing dependencies. This was problematic because it caused version mismatches to occur.

Here I will describe a solution I've come up with for how we can do this as a team. It is not clean, but it should work for our purposes. 

### Committing changes directly from Colab

When you open Colab, the default screen shows you your 'Recents'. Navigate to the 'GitHub' tab and then check the 'Include private repos' box. Then in the search bar, find 'namiyousef/argument-mining', and select the branch to be 'develop'. At some point, you will be asked to connect your account to Github. Make sure you do so when prompted.

Once that is done, you'll be able to see all the notebooks in the repository in the last state that they were pushed.

These notebooks don't exist locally, so if you make changes and try to save them there'll be a pop-up of "don't have permissions to save notebook, save in local drive". Don't save it locally. Instead, click on File->Save a copy in GitHub (NOT GitHub Gist). Update the commit message to something appropriate and submit :)

### Syncing the repo with Google Drive

There are multiple ways that this can be done, however to keep things simple I will go through a single method only. This assumes that you've already cloned your repository locally.

First, download Google Drive for your computer. Once this has been done, sync the local copy of your repo with Google Drive. This will save the repository under 'Other Computers' within Google Drive. Now go into the broswer version of Google Drive, find the repository folder, right click and then click 'add shortcut to drive'. This will make sure that the folder is accessible through 'My Drive'.

### Installing the virtual environment

In order to make things easy, I've written some code that automatically installs ArgMiner as a package in your Colab. Please find the relevant code in `experiments/yousef/test.ipynb`

### Pointers

- To avoid issues, please make sure that your repository is always updated (e.g. using `git fetch`) and that local changes are fully synced to Google Drive before you run notebooks on it.
- If you update your notebooks on Colab, then your local repository will be out of sync. Make sure you fetch before persisting any local changes

For now, it is not possible to save things into Colab in a clear and consistent way. So if you want to save anything please do it locally. When we play around with this setup further I'll find a way of saving things nicely.

## Making changes to ArgMiner
I've laid out the project as such so that we don't clutter the actual package ArgMiner so much. The idea is that we can keep experimenting in our own folders under `experiments`. When we are ready to persist new changes to argminer, we can review it together. As of right now, this will be done in an ad-hoc fashion. There are no automated tests just yet.

## Tests
