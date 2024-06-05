# From A to B with RL !

Hello !

## Setup

We work with the `rl` virtual environment, based on Python v3.11.9, so let us
first install that python version :

```
  pyenv install 3.11.9
```

If you don't have pyenv, please [install it first](https://github.com/pyenv/pyenv).
If you are on Windows, you can install [pyenv-win](https://pyenv-win.github.io/pyenv-win/)

You then have to create a virtual environment named `rl` :

```
  pyenv virtualenv 3.11.9 rl
````

Then, you can cd to the `from-a-to-b-with-rl` folder, and check if it is activated.

If the virtual environment is not activated automatically upon entering the folder you can run:

```
  pyenv local rl
````

### Packages

The `rl` virtual environment has a few dependencies, notably :

- [pytorch](https://pytorch.org/) for the RL learning
- [numpy](https://numpy.org/) to handle a few numerical outputs
- [gymnasium](https://gymnasium.farama.org/)
- [pygame](https://www.pygame.org/news)
- [moviepy](https://pypi.org/project/moviepy/) to save video from the agent interacting with the environment

You can then decide to install the package itself (XXX
Note, for now, nothing interesting is installed except from the dependencies XXX):

```
  pip install .
````

Or just decide to install the `requirements.txt` file :

```
  pip install -r requirements.txt
```

### Accessing a remote computer using SSH

Even if you are on windows, normally you _should_ have `ssh-keygen` and `ssh` available
as commands from the prompt.

First, let's create a SSH key using the [ed25519](https://en.wikipedia.org/wiki/EdDSA#Ed25519)
algorithm :

```
  ssh-keygen -t ed25519
```

It should normally ask you to enter a file name (see below typical output).
No bad choices here unless a file already exists, you can call it `anything`.

```
  Generating public/private ed25519 key pair.
  Enter file in which to save the key (/Users/indrianylionggo/.ssh/id_ed25519): .ssh/anything
  Enter passphrase (empty for no passphrase):
```

It actually creates _two_ files, one being `anything` (your _private_ key) and
one being  `anything.pub`, both in your `~/.ssh/` folder. **Keep your private
key private !** and send the public key to Brice through Slack or anything.

Then, you can ask Brice to connect to his home desktop. I will not put my public
IP here since I am not crazy. It is available on the Slack channel. Let's say my
IP address is `XXX.XXX.XXX.XXX` :

```
  ssh lewagon@XXX.XXX.XXX.XXX
```

If Brice has done his job, your ssh authentication key will automatically be
recognized. However, your computer will not be sure of where it connects, so it
could throw a warning :

```
  The authenticity of host 'github.com (IP ADDRESS)' can't be established.
  ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
  Are you sure you want to continue connecting (yes/no)?
```

Type `yes`. You might be asked for either :

- the password for your private SSH key : only you know it.
- the password for the session `lewagon` : I have put it on our Slack channel.

Input them, then normally you should see :

```
  C:/users/lewagon/>
```

My machine is a Windows, so some Linux commands will not work (`touch`, maybe even `ls`). But

* python should work : `python --version`
* pyenv should work : `pyenv --help`
* git should work : `git --help`

Other interesting commands are `dir` (print folder contents) and `cd`. The project folder is located at : 

```
  C:/users/lewagon/python/from-a-to-b-with-rl
```

#### On VSCode

Once you have checked that the connection works, you can try connecting _through_ VSCode. Click on the blue box (see below) and select `connect to host`, then ``

![img](vscode_ssh_screenshot.png)

Add a new ssh host and follow the instructions of the box that pops up, re-type the command :

```
  ssh lewagon@XXX.XXX.XXX.XXX
```
VSCode might ask you to save some changes to a config file so you don't have to retype the ssh command again. You can say yes to that. Then, it will probably fail once after asking for the remote computer password. Just specify the remote machine is
a Windows, and try again, normally it should work !

**/!\\** The local account has a lot of privileges, __but__ you cannot access github from it ! So if you need to `git pull`, you will have
to ask Brice to do it for you (he has a GitHub tunnel). Also : don't clutter the computer.

### Notes on GPU acceleration :

If your GPU is CUDA capable, you will have to adapt your `rl` environment. If you are on Windows, you can type :

```
  pip uninstall torch

  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you are on Linux, you can do :

```
  pip uninstall torch

  pip3 install torch torchvision torchaudio
```
