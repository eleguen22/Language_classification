# Wikipedia Language Classification

### Project : Language Classification using Decision Tree and Adaboost

### Introduction :
The goal of this project is to classify two languages by using either a Decision Tree algorithm or 
Adaboost algorithm (boosted decision stump). In our use case we classify "English" and "Dutch" and we 
provide the dataset for the training and for the test step. We can also classify other languages 
by changing the data and the features of the sentence class.

### Decision tree Algorithm :

We first use a decision tree  to classify the data. We also use the metric of information gain to 
split our data  by features and build our decision tree. We use the entropy in order to measure
the disorder in our dataset and then compute our information gain.

### Adaboost :

We create a set of boosted stumps (decision tree of depth equals to 1) in order to classify the data. Each 
sentence of the dataset has a weight. We adjust this weight with the result of the classification. We also
boost each stump with a weight by using the amount of say.

# Utilisation :

We can use this project either in a Train mode or in a Predict mode as below :

###Train step :

<b>python sort.py train</b> `<examples>` `<hypothesisOut>` `<learning-type>`
- `<examples>` is a file containing labeled sentences.
- `<hypothesisOut>` specifies the filename to write your model to.
- `<learning-type>` specifies the type of learning algorithm you will run, it is either "dt" or "ada".

###Prediction step :

<b>python sort.py predict</b> `<hypothesis>` `<file>`
 - `<hypothesis>` is a pre-trained model created by the train program
 - `<file>` is a file containing test data.

###Format for Training data :

```
<label>|<data>
```
For English and Dutch the labels are ```en``` and ```nl``` respectively.
###Format for the Test data :
For each sentence of the Test file we return the prediction of the algorithm and we print this
prediction line by line.

```
<data>
```

###Dataset

The data for the training and the test are in the `/data_in/train_dataset.dat` and  `/data_in/test_dataset.dat`directory.

### Pre-trained Models
Trained models that classify English and Dutch can be found in the `\out` directory.

 - `\model_out\model_adaboost.oj` 
 - `\out\model_tree.oj` 
 
You could run this model in order to classify data with the function predict (Prediction Step).

**PS :** You can change the features especially if you want to classify others languages. 