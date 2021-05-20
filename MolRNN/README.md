# Description of the project

[Project]()

## How to start

The packages required for the execution of the code along with the role of each file and the software used are described
in the Sections below.

## Setup

1. Install all the packages appointed in
   the [requirements.txt]() file.

2. Download the project directory
   from [GitHub]().
   
7. The installation of rdkit needs an additional step. Firstly, run the following line on the terminal:
   ```
   conda install -c rdkit rdkit
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
   
3. The project is developed on mxnet 1.3.1 and python 3.6. To work on GPUs you must install both Cuda 9.2 and
   cudnn 7.6.5.

## Run the code

Once all the necessary packages have been installed you can run the code by typing this line on the terminal or by the
specific command within the IDE.

```
python train.py
```

## Role of each file

**main.py** is the starting point of the entire project. It defines the order in which instructions are realised. More
precisely, it is responsible to call functions from other files in order to divide the datasets provided, pre-process
images as well as to instantiate, train and test the models.

**config.py** makes available all the global variables used in the project.

**utilities.py** includes functions to download and split the datasets in the dedicated folder, to compute the mean RGB
value of the dataset and to plot results.

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is an integrated development environment (IDE) for Python programmers: it was chosen because it is one of the
most advanced working environments and for its ease of use.

> <img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" width="140" alt="colab">

Google Colab is an environment that enables to run python notebook entirely in the cloud. It supports many popular
machine learning libraries and it offers GPUs where you can execute the code as well.

> <img src="https://user-images.githubusercontent.com/674621/71187801-14e60a80-2280-11ea-94c9-e56576f76baf.png" width="80" alt="vscode">

Visual Studio Code is a code editor optimized for building and debugging modern web and cloud applications.
