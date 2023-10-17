# Kaggle Time Series Forecasting Software Package

## Purpose

This project is designed to serve three main purposes:

1. Showcase the basics of feature generation and extraction within the dataset archetype of time series
2. Create a framework for creating, training, and evaluating models using base or featured time series dataset from Kaggle Store Sales Time Series Forecasting
3. Utilize clean code and software engineering practices to enable ease of use and editing

## Project Description

Kaggle hosts a variety of machine learning problems that are designed as competitions to be solved by a large community. However, some competitions are designed to instead help introduce programmers to concepts new and old within the broad field. One such competition is the "Store Sales Time Series Forecasting Challenge", a competition with the goal of teaching programmers to use time series data to create machine learning models that predict future data, bundled with a real world application. The scope of this project is to showcase my research into the data type of time series, and how to solve machine learning problems within that type while using modern clean code and software development practices.

## Index

1. Dependencies
2. Installation
3. Research Usage
4. Software Usage
5. Troubleshooting
6. Licensing

## Dependencies

All dependencies that were used by the owner are captured within the requirements.txt file, which is handled within the Installation section. However, it is important to note that they used VSCode as the IDE, and thus used the Windows OS and Powershell/VSCode terminal to perform the following operations.

## Installation

1) To download the github repository, simply navigate to the folder you want to use and type this command in git bash:
```python git clone https://github.com/jakenorbie12/cuddly-computing-machine.git```

2) To install dependences, in anaconda navigate to the directory and type:
```pip install --file requirements.txt```

## Use Kaggle Dataset

1) In order to download the Kaggle dataset you can either do so at `https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data` or by using the Kaggle API

2) To use Kaggle API you must first verify your token at your account page and selecting the option 'Create New Token' under API. This will download the kaggle.json file

3) Move the file into the folder of C://Users/*user*/.kaggle

4) Use this command to download the dataset
```kaggle competitions download -c store-sales-time-series-forecasting -p ./data/original```

5) Unzip the folder. Using powershell the command is:
```Expand-Archive ./data/original/store-sales-time-series-forecasting.zip -DestinationPath ./data/original```

## Research

Firstly, the research uses a common format of Jupyter Notebooks. Since this was used in VSCode, you will need to go to the Command Palette (Ctrl + Shift + P) and create a python environment. This project uses a venv and uses the requirements.txt file to import all necessary imports.

## Software

Currently there is no usage, other than to view the research work into time series. Be on the lookout for a deployable package update!

## Troubleshooting

For any issues you may have, please feel free to email me at `jakenorbie@gmail.com`.

## Licensing

Jake Norbie
