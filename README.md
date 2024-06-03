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

### Packages

The `rl` virtual environment has a few dependencies, notably :

- [pytorch](https://pytorch.org/) for the RL learning
- [numpy](https://numpy.org/) to handle a few numerical outputs
- [gymnasium](https://gymnasium.farama.org/)
- [pygame](https://www.pygame.org/news)

You can then decide to install the package itself (XXX
Note, for now, nothing interesting is installed except from the dependencies XXX):

```
  pip install .
````

Or just decide to install the `requirements.txt` file :

```
  pip install -r requirements.txt
```

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
