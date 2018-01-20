# Dependency_Parser
This project was written by Carmel Rabinovitz and Amir Livne as part of NLP Technion course
This is a dependency parser tagger (using max entropy).
We reached 85.36% accuracy over test set
 
## How to use:
To run the Dependency Parser on comp data use:
```
python main.py <your results directory name> -comp_path <path to comp> -train_path <path to train data> -test_path <path to validation data>
```

if -comp_path is no provided it will defualtly use:
current script directory + '\\data\\comp.unlabeled'

if -train_path is no provided it will defualtly use:
current script directory + '\\data\\train.labeled'

if -test_path is no provided it will defualtly use:
current script directory + '\\data\\test.labeled'
	
## Results:
The script will create a directory named "results directory name" containing the results files:
  1. model.pkl - the Dependency Parser model
  2. predictions.wtag - predictions of the comp data provided
  3. logs.txt - information about the training congifurations

additional flags:
1. -max_iter <#>, 50 is default value
2. -mode <base, complex>, complex is default value
