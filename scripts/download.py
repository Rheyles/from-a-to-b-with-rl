import os

# NOTE : Leave model blank to list the available files
model = '0607_1707_CarDQNAgent'
# model = ''

# Do not change these
or_folder = './python/from-a-to-b-with-rl/models/'

if model != '':
    print('Downloading from destination')
    dest_folder = './models/'
    command = f'scp -r lewagon@82.67.97.37:{or_folder + model} {dest_folder + model}'
    os.system(command)

else:
    print('Listing files in destination')
    command = f'ssh lewagon@82.67.97.37'
    os.system(command)
    # Then
    # > cd python
    # > cd from-a-to-b-with-rl
    # > cd models
    # > dir
    # and check the files
