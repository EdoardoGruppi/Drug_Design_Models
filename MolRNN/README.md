# Description of the project

[Project](https://github.com/EdoardoGruppi/MolRNN)

This project is a reimplementation of the models introduced in the following paper: "Multi-objective de novo drug design
with conditional graph generative model" ([paper](https://link.springer.com/content/pdf/10.1186/s13321-018-0287-6.pdf)).
Specifically, the code is an updated version of that published by the authors in
this [repository](https://github.com/kevinid/molecule_generator). The code now works with python 3.6 and also allows
those who work on windows to install certain versions of some libraries necessary to run the program. Furthermore, some
bugs are also found and solved. Most importantly, the code presented in this project is widely commented to facilitate
the reading.

**Important:** Even if the code presented in this repository is almost entirely based on the code published by the
authors in their [repository](https://github.com/kevinid/molecule_generator) the results might differ for some reason.
Therefore, for any benchmark test to be performed on the models of the paper, please refer to the original code.

## How to start

The packages required for the execution of the code along with the role of each file and the software used are described
in the Sections below.

## Setup

<!--
Code converted running the following command on the terminal:
python -m lib2to3 --output-dir=C:\Users\<user>\<path>\<folder-where-to-save-py3-code> -W -n C:\Users\<user>\<path>\<folder-with-py2-code>
-->

1. Download the project directory from [GitHub](https://github.com/EdoardoGruppi/MolRNN).

2. Install all the packages appointed in
   the [requirements.txt](https://github.com/EdoardoGruppi/MolRNN/blob/main/requirements.txt) file or follow the below
   steps.

3. Install the correct rdkit version with conda:

   ```
   conda install -c rdkit rdkit==2018.03.3.0
   ```

   **Important:** Finally, since some experimental functions not directly included in the package are used, it is
   necessary to copy and paste the Contrib folder from:

   ```
   C:\Users\<'name user'>\anaconda3\pkgs\<'rdkit version'>\Library\share\RDKit
   ```

   to:

   ```
   C:\Users\<'name user'>\anaconda3\envs\<'name env'>\Lib\site-packages\rdkit
   ```

4. The project is developed on mxnet 1.3.1 and python 3.6. To work on GPUs you must install both Cuda 9.2 and cudnn
   7.6.5. You can directly use conda.

5. Install the correct version of mxnet:

   ```
   pip install mxnet-cu92==1.3.1
   ```

6. Resolve the dependency conflicts downgrading the versions of some libraries such as pandas and mkl_fft.

7. Install the other packages required if not already installed:

   ```
   pip install pandas scipy matplotlib networkx protobuf sklearn lmdb pymysql
   ```

## Run the code

Once all the necessary packages have been installed you can run the code by typing this line on the terminal or by the
specific command within the IDE.

```
python main.py
```

## Role of each file

**main.py** is the starting point of the entire project.

**train.py** defines the order in which instructions are realised. More precisely, it is responsible to call functions
from other files in order to divide the datasets provided, pre-process data as well as to instantiate, train and test
the models.

**test.py** script to test the models already trained and saved.

## Issues

1. The code is focused on testing the MolRNN model. It is thus necessary to add some lines to run the MolMP model.

2. An error occurs when num_workers is greater than zero. Do not change the parameter value to other values.

## References

Li, Yibo, Liangren Zhang, and Zhenming Liu. "Multi-objective de novo drug design with conditional graph generative
model." Journal of cheminformatics 10.1 (2018): 1-24.

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is an integrated development environment (IDE) for Python programmers: it was chosen because it is one of the
most advanced working environments and for its ease of use.

> <img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" width="140" alt="colab">

Google Colab is an environment that enables to run python notebook entirely in the cloud. It supports many popular
machine learning libraries and it offers GPUs where you can execute the code as well.

> <img src="https://user-images.githubusercontent.com/674621/71187801-14e60a80-2280-11ea-94c9-e56576f76baf.png" width="80" alt="vscode">

Visual Studio Code is a code editor optimized for building and debugging modern web and cloud applications.
