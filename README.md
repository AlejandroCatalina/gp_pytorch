# GP PyTorch : Reimplementing Gaussian processes in Pytorch

## To use the package:
  1. Create a conda environment or a python virtual environment.
  2. Activate the conda / virtual environment.
  3. `git clone https://github.com/AlejandroCatalina/gp_pytorch.git`
  4. `cd gp_pytorch`
  5. `python -m pip install -e .`
  
## To run the included example:
  1. `python examples/simple_regression.py`
  
## To use the modules in your own GP implementation
  1. Install the package as described in *To use the package*.
  2. You can now access the models as:
  ```python
  from gppytorch.models import GPR, SGPR, DGP
  ```
  3. The kernels and lossess can similarly be accessed as:
  ```python
  from gppytorch.kernels import SquaredExp
  from gppytorch.losses import elbo
  ```
  
## Roadmap

  - [x] Exact GP implementation and testing.
  - [x] Sparse GP implementation and testing.
  - [x] Deep GP implementation and testing.
  - [x] Flow GP implementation.
  - [ ] Flow GP testing. 
  It needs more testing, it seems to work but we have yet to fully understand it so we can develop some intuition about how many samples and integration steps are sensible defaults.
