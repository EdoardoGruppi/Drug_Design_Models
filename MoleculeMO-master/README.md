# Description of the project

This project is a reimplementation of the model introduced in the following paper: "Multiobjective de novo drug design with recurrent neural networks and nondominated sorting." ([paper](https://link.springer.com/article/10.1186/s13321-020-00419-6)). Specifically, the code is a slightly updated version of that published by the author in this [repository](https://github.com/jyasonik/MoleculeMO).

**Important:** Even if the code presented in this repository is almost entirely based on the code published by the
authors in their [repository](https://github.com/jyasonik/MoleculeMO) the results might differ for some
reason. Therefore, for any benchmark test to be performed on the models of the paper, please refer to the original code.

## How to start

The packages required for the execution of the code are described in the [requirements.txt](https://github.com/EdoardoGruppi/Drug_Design_Models/blob/main/MoleculeMO-master/requirements.txt). The packages can alternatively be installed following the first commands written in the IPython Notebook [Script_Yasonik.ipynb](https://github.com/EdoardoGruppi/Drug_Design_Models/blob/main/MoleculeMO-master/Script_Yasonik.ipynb). The latter enable to directly run the code on platforms such as Google Colab, Jupyter Notebooks or Kaggle.

## Run the code

Once all the necessary packages have been installed you can run the code by typing this line on the terminal or by the
specific command within the IDE.

```
python main.py
```

Additionally, the code can be executed running cell by cell the aforementioned notebook.

## References

Yasonik, Jacob. "Multiobjective de novo drug design with recurrent neural networks and nondominated sorting." [paper](https://link.springer.com/article/10.1186/s13321-020-00419-6) Journal of Cheminformatics 12.1 (2020): 1-9.

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is an integrated development environment (IDE) for Python programmers: it was chosen because it is one of the
most advanced working environments and for its ease of use.

> <img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" width="140" alt="colab">

Google Colab is an environment that enables to run python notebook entirely in the cloud. It supports many popular
machine learning libraries and it offers GPUs where you can execute the code as well.

> <img src="https://user-images.githubusercontent.com/674621/71187801-14e60a80-2280-11ea-94c9-e56576f76baf.png" width="80" alt="vscode">

Visual Studio Code is a code editor optimized for building and debugging modern web and cloud applications.
