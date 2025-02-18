
# Corporate Social Responsibility via Multi-Armed Bandits

This repository is the official implementation of [Corporate Social Responsibility via Multi-Armed Bandits](https://github.com/tomron/CSRviaMAB/blob/master/CSRviaMAB.pdf). 

>We propose a multi-armed bandit setting where each arm corresponds to a subpopulation, and pulling an arm is equivalent to granting an opportunity to this subpopulation. In this setting the decision-maker's fairness policy governs the number of opportunities each subpopulation should receive, which typically depends on the (unknown) reward from granting an opportunity to this subpopulation. The decision-maker can decide whether to provide these opportunities or pay a pre-defined monetary value for every withheld opportunity. The decision-maker's objective is to maximize her utility, which is the sum of rewards minus the cost of withheld opportunities. We provide a no-regret algorithm that maximizes the decision-maker's utility and complement our analysis with an almost-tight lower bound. Finally, we discuss the fairness policy and demonstrate its downstream implications on the utility and opportunities via simulations.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Run simulations in the paper

To run the simulations in the paper, run this command:

```simulations
python runSimulations.py --output_dir <output_dir>
```

> When running a simulation with output directory `output`, it is assumed that there exists a directory `output/raw`


## Run a single simulation

To run a single simulation with your own parameters, run this command:

```single simulation
python BlackBoxAlgorithm.py --arms 0.2 0.35 0.5 0.65 0.8 0.95 --output_dir output --T 1000 --lambda 0.4
```

Full usage - 
```
usage: BlackBoxAlgorithm.py [-h] [--lambda LAM] [--repeat REPEAT] [--T T]
                            --arms ARMS [ARMS ...] [--output_dir OUTPUT_DIR]
                            [--fairness_function {lin,const,softmax}]
                            [--fairness_portion FAIRNESS_PORTION]

optional arguments:
  -h, --help            show this help message and exit
  --lambda LAM          penalty term
  --repeat REPEAT       how many times to repeat the experiment
  --T T                 horizon
  --arms ARMS [ARMS ...]
                        Arms probabilities
  --output_dir OUTPUT_DIR
                        output directory
  --fairness_function {lin,const,softmax}
                        fairness function
  --fairness_portion FAIRNESS_PORTION
                        multiplier of the fairness function
```

## Evaluation

To create a summary of results run -

```analyze
python AnalyzeResults.py --input_dir <input_dir> --output_dir <output_dir>
```

The input dir used here is the output directory for the simulation.

To create graphs an in the paper run -
```graph
python AnalyzeResults.py --input_dir <input_dir> --output_dir <output_dir>
```

Input dir is the output directory of the AnalyzeResults.py script.
PDF files would be created in output_dir along with the log and aux files.

```
usage: GenerateGraphs.py [-h] [--input_dir INPUT_DIR]
                         [--output_dir OUTPUT_DIR]
                         [--templates_dir TEMPLATES_DIR]

Generate graphs

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Simulation input directory
  --output_dir OUTPUT_DIR
                        pdf files output directory
  --templates_dir TEMPLATES_DIR
                        Tex files template directory
```


## Results

Average utility per round for different transfer costs (0, 0.4, 0.8) and different fairness functions - uniform, linear and softmax.
Each R-O instance had six arms (K=6) with expected reward - 0.2, 0.35, 0.5, 0.65, 0.8 and 0.95 and varying horizons - 1K, 5K, 10K, 50K, 100K and 200K. Each experiment was executed 200 times. We used UCB1 as the black box algorithm _ALG_ in  _Self-regulated Utility Maximization_.

| transfer cost = 0         | transfer cost = 0.4  | transfer cost = 0.8 |
| ------------------ |---------------- | -------------- |
| ![lam_0.0](/resources/lam_0.0.jpg)   |     ![lam_0.4](/resources/lam_0.4.jpg)      |     ![lam_0.8](/resources/lam_0.8.jpg)       |


Round distributions per phase for softmax
| transfer cost = 0         | transfer cost = 0.4  | transfer cost = 0.8 |
| ------------------ |---------------- | -------------- |
| ![lam_0.0](/resources/pull_dist_0.0_Softmax.jpg)   |     ![lam_0.4](/resources/pull_dist_0.4_Softmax.jpg)      |     ![lam_0.8](/resources/pull_dist_0.8_Softmax.jpg)       |


## Contributing

Licence - Apache License, Version 2.0, January 2004

Contributions are welcome. Please open a pull request and we check it and merge it if relevant.
