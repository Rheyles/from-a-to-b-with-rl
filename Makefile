map = 4x4# Frozen Lake Maps : 4x4, 5x5_easy, 5x5_hard, 8x8
continuous = false# For LunarLander, MountainCar, CarRace

# Frozen Lake
frozenlake:
	cp code/frozenlake/params_${map}.py code/frozenlake/params.py
	python code/frozenlake/DQN_frozen_lake.py
	python code/frozenlake/DQN_frozen_lake.py --eval ${map}

frozenlake-train:
	cp code/frozenlake/params_${map}.py code/frozenlake/params.py
	python code/frozenlake/DQN_frozen_lake.py --train

frozenlake-eval:
	python code/frozenlake/DQN_frozen_lake.py --eval ${map}
	
frozenlake-eval-video:
	python code/frozenlake/DQN_frozen_lake.py --eval ${map} --video
	
# Cartpole
cartpole:
	python code/cartpole/A2C_cartpole.py --train
	python code/cartpole/A2C_cartpole.py --eval

cartpole-train:
	python code/cartpole/A2C_cartpole.py --train

cartpole-eval:
	python code/cartpole/A2C_cartpole.py --eval

cartpole-eval-video:
	python code/cartpole/A2C_cartpole.py --eval --video

# LunarLander
lunarlander:
ifeq ($(continuous),true)
	python code/lunarlander/A2C_lunarlander_continuous.py --train
	python code/lunarlander/A2C_lunarlander_continuous.py --eval
else
	@echo /!\\ Discrete Lunar Lander. Add 'continuous=true' at the end of the make command if you do not want a continuous model
	python code/lunarlander/A2C_lunarlander.py --train
	python code/lunarlander/A2C_lunarlander.py --eval
endif

lunarlander-train:
ifeq ($(continuous),true)
	python code/lunarlander/A2C_lunarlander_continuous.py --train
else
	@echo /!\\ Discrete Lunar Lander. Add 'continuous=true' at the end of the make command if you do not want a continuous model
	python code/lunarlander/A2C_lunarlander.py --train
endif

lunarlander-eval:
ifeq ($(continuous),true)
	python code/lunarlander/A2C_lunarlander_continuous.py --eval
else
	@echo /!\\ Discrete Lunar Lander. Add 'continuous=true' at the end of the make command if you do not want a continuous model	
	python code/lunarlander/A2C_lunarlander.py --eval
endif

lunarlander-eval-video:
ifeq ($(continuous),true)
	python code/lunarlander/A2C_lunarlander_continuous.py --eval --video
else
	@echo /!\\ Discrete Lunar Lander. Add 'continuous=true' at the end of the make command if you do not want a continuous model
	python code/lunarlander/A2C_lunarlander.py --eval --video
endif


# CarRacing
carracing:
ifeq ($(continuous),true)
	python code/carracing/DQN_carracing_continuous.py --train
	python code/carracing/DQN_carracing_continuous.py --eval
else
	@echo /!\\ Discrete CarRacing. Add 'continuous=true' at the end of the make command if you do not want a continuous model
	python code/carracing/DQN_carracing.py --train
	python code/carracing/DQN_carracing.py --eval
endif

carracing-train:
ifeq ($(continuous),true)
	python code/carracing/DQN_carracing_continuous.py --train
else
	@echo /!\\ Discrete CarRacing. Add 'continuous=true' at the end of the make command if you do not want a continuous model
	python code/carracing/DQN_carracing.py --train
endif

carracing-eval:
ifeq ($(continuous),true)
	python code/carracing/DQN_carracing_continuous.py --eval
else
	@echo /!\\ Discrete CarRacing. Add 'continuous=true' at the end of the make command if you do not want a continuous model	
	python code/carracing/DQN_carracing.py --eval
endif

carracing-eval-video:
ifeq ($(continuous),true)
	python code/carracing/DQN_carracing_continuous.py --eval --video
else
	@echo /!\\ Discrete CarRacing. Add 'continuous=true' at the end of the make command if you do not want a continuous model
	python code/carracing/DQN_carracing.py --eval --video
endif