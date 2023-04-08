# Generalized Forecasting Framework for Financial Paradigms
This repository contains the code, data, and resources used in our research paper, "A Generalized Forecasting Framework for Analyzing Domain-Dependent Performance of Financial Paradigms". Our paper investigates the theoretical lower bound loss of various forecasting paradigms and demonstrates their domain-dependent accuracy through a generalized forecasting framework.

## Abstract
In this paper, we present a total ordering of the theoretical lower bound loss of different forecasting paradigms in the following descending order: Model selection, Model combination, Non-parametric univariate models, and Non-parametric multivariate models. We then create a generalized forecasting framework to test these paradigms ex-ante. The framework is implemented using a novel datacube that consists of daily stock prices, 100,000 quarterly reports from about 1600 global companies, and several daily macro time series, spanning from 2000 to spring 2022. Utilizing this framework, we show that modern multivariate time series approaches are powerful but domain-dependent. We demonstrate the domain-dependent accuracy by illustrating convincing results when predicting corporate bankruptcy risk, moderate results when predicting stock price volatility, and lacking results when predicting company market capitalization. Given the domain-dependent convincing results and mostly unrealized theoretical lower bound loss of multivariate approaches, we hope to encourage further research on non-parametric, multi-signal approaches that leverage a wider array of available information.

## Repository Structure
code/: Contains all the code and scripts used for implementing the forecasting framework, data preprocessing, and model evaluation.
data/: Contains the datacube with daily stock prices, quarterly reports, and macro time series data. Note: Due to the size of the dataset, you may need to download the data separately (see instructions below).
models/: Contains pre-trained models and results from our experiments.
paper/: Contains the full-text version of our research paper, along with supplementary materials.
docs/: Contains documentation and guidelines for using our generalized forecasting framework.
Getting Started
Clone the repository: Clone this repository to your local machine using git clone https://github.com/Krankile/npmf.git.

Download the data: Due to the size of the dataset, you will need to download it separately. Follow the instructions in data/README.md to obtain the data.

Install dependencies: Install the required Python libraries by running pip install -r requirements.txt.

Run the experiments: Navigate to the code/ directory and follow the instructions in code/README.md to run the experiments and reproduce our results.

Explore the paper: Read the full-text version of our research paper in the paper/ directory, and delve into supplementary materials for a deeper understanding of our methodology and findings.

## Contributing
We welcome contributions from the community to help improve our generalized forecasting framework and expand its applications. Please read the CONTRIBUTING.md file for more details on how to contribute to this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use our code, data, or findings in your research, please cite our paper as follows:

```latex
@mastersthesis{ankile2022exploration,
  title={Exploration of Forecasting Paradigms and a Generalized Forecasting Framework},
  author={Ankile, Lars Lien and Krange, Kjartan},
  year={2022},
  school={NTNU}
}
```
