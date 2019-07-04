# Real Time Inference

Set up your environment using `environment.yml`. For Conda run: 

`conda env create -f environment.yml`

The name of the environment will be the first line in the .yml file. 

If that fails you, simply `conda install` the following packages:
- opencv
- numpy
- requests

Activate your environment, and start the server with:

`python server.py`
