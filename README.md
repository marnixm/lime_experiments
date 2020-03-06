## M. Maas (2020). How do you explain that? An assessment of black box model explainers.

Credit is given to the original creators of the experiment from the [Lime paper](http://arxiv.org/abs/1602.04938). If you're interested in using LIME, check out [this repository](https://github.com/marcotcr/lime).
Shap is implemented in [this](https://github.com/slundberg/shap) package.

Requirements for the experiments of this study can be found in file:
`requirements.txt`

For any questions regarding these experiments, please contact [me](mailto:marnixmaas@live.nl?subject=[Master%20thesis]).

## Experiment 5.2:
To run this experiment, use `run_evaluate.py`. All parameter are specified on the top of the file. 
1. DATASET -> 'multi_polarity_books', 'multi_polarity_kitchen', 'multi_polarity_dvd'
2. ALGORITHM -> 'l1logreg', 'tree'
3. EXPLAINER -> 'Shap', 'lime', 'parzen'

To run the experiment, use function `run_5_2()` on the bottom of the script.

To show the results of the experiment, use function `plot_5_2()` and give a results file as parameter.

## Experiment 5.3:
To run this experiment, use `run_trust.py`. All parameter are specified on the top of the file. 
1. DATASET -> 'multi_polarity_books', 'multi_polarity_kitchen', 'multi_polarity_dvd'
2. ALGORITHM -> 'logreg', 'Nearest neighbours', 'random_forest', 'svm', 'tree'
3. EXPLAINER -> 'Shap', 'lime', 'parzen'

To run the experiment, use function `run_5_3()` on the bottom of the script.

To show the results of the experiment, use function `table_5_3()`. Choose either statistic, or the adjusted F1 score.

## Multi-polarity datasets:
I got them from [here](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/). Second dataset: [processed_acl.tar.gz]

## About the author
This research was done by Marnix Maas as Thesis for:
* Master Business Analytics at the VU University Amsterdam
* Accenture Netherlands.
 
For bio see [LinkedIn](https://www.linkedin.com/in/marnixwmaas/).

 

Supervisor from exact sciences department: Sandjai Bhulai.

