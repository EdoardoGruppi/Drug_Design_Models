# Description of the project

[Project](https://github.com/EdoardoGruppi/Drug_Design_Models/tree/main/GCPN)

This project is a reimplementation of the model introduced in the following paper: "Graph convolutional policy network
for goal-directed molecular graph generation." ([paper](https://arxiv.org/pdf/1806.02473.pdf)). Specifically, the code
is a slightly updated version of that published by the authors in
this [repository](https://github.com/bowenliu16/rl_graph_generation)
. The code now works with tensorflow 2. The code presented in this project is commented to facilitate the reading.

**Important:** Even if the code presented in this repository is almost entirely based on the code published by the
authors in their [repository](https://github.com/bowenliu16/rl_graph_generation) the results might differ for some
reason. Therefore, for any benchmark test to be performed on the models of the paper, please refer to the original code.

## How to start

The packages required for the execution of the code along with the role of each file and the software used are described
in the Sections below.

## Setup

<!--
comment the line that install the gym package in the setup.py of the baselines
-->

1. Download the project directory from [GitHub](https://github.com/EdoardoGruppi/Drug_Design_Models/tree/main/GCPN).

2. Install all the packages appointed in
   the [requirements.txt](https://github.com/EdoardoGruppi/Drug_Design_Models/blob/main/GCPN/requirements.txt) file or follow the below
   steps.

3. Run the following lines on the terminal:

   ```
   conda install -c conda-forge mpi4py
   conda install networkx==1.11
   ```

4. Change directory and install all the packages defined in the rl-baselines setup.py:

   ```
   cd rl-baselines
   pip install -e.
   ```

5. Change directory and install all the packages defined in the gym-molecule setup.py:

   ```
   cd gym-molecule
   pip install -e.
   ```

6. Install the following libraries with pip:

   ```
   pip install matplotlib tensorboardX
   ```

7. Install the rdkit package with the following command>

   ```
   conda install -c rdkit rdkit
   ```

8. The code works on GPU as soon as the proper versions of cuda and cudnn are installed. Install cuda toolkit running
   the below line in the terminal:

   ```
   conda install -c anaconda cudatoolkit=11.0
   ```

   Then install with the same modalities the cudnn package required.

9. Insert the following code in the main.py to allocate the GPU memory only when effectively needed.
   ```
   physical_devices = tf.config.experimental.list_physical_devices('GPU')
   print("Num GPUs Available: ", len(physical_devices))
   if len(physical_devices) is not 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)
   ```

## Run the code

Once all the necessary packages have been installed you can run the code by typing this line on the terminal or by the
specific command within the IDE.

```
python main.py
```

## Role of each file

**main.py** is the starting point of the entire project.

## References

You, Jiaxuan, et al. "Graph convolutional policy network for goal-directed molecular graph
generation." [arXiv](https://arxiv.org/abs/1806.02473) preprint arXiv:1806.02473 (2018).

## Issues

- if a problem occurs during the installation of cuda toolkit and cudnn, be sure to have installed the proper versions
  of the libraries. Note: the version depends on the device used to run the code. In case the error outputted state that no library
  cusolver64_11.dll is found, rename the cusolver64_10.dll file to cusolver64_11.dll in the folder:

  ```
   C:\Users\<user name>\anaconda3\envs\<name env>\Library\bin
  ```

- if the following error interrupts the code execution, please ensure that the mpi4py is installed as described above.
  ```
  from mpi4py import MPI ImportError: DLL load failed:
  ```

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is an integrated development environment (IDE) for Python programmers: it was chosen because it is one of the
most advanced working environments and for its ease of use.

> <img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" width="140" alt="colab">

Google Colab is an environment that enables to run python notebook entirely in the cloud. It supports many popular
machine learning libraries and it offers GPUs where you can execute the code as well.

> <img src="https://user-images.githubusercontent.com/674621/71187801-14e60a80-2280-11ea-94c9-e56576f76baf.png" width="80" alt="vscode">

Visual Studio Code is a code editor optimized for building and debugging modern web and cloud applications.
