# Determining weights to fit a linear regression model
The program  implements a linear regression to fit a line or a second-degree polynomial to a set of training data.
The program can be invoked as follows:
linear_regression <training_file> <degree> <?>
The arguments provide to the program the following information:

-The first argument is the path name of the training file, where the training data is stored.
The path name can specify any file stored on the local computer.
-The second argument is a number. This number should be either 1 or 2. 
 If the number is 1, it fits a line to the data. If the number is 2, it  fits a second-degree polynomial to the data.
-The third number is a non-negative real number (it can be zero or greater than zero). 
This is the value of ? that is used for regularization. If ? = 0, then no regularization is done.

The training data is of the format as in sample_data1.txt
## Linear model 
![image](https://user-images.githubusercontent.com/26680128/33006414-e7ab0c60-cd8f-11e7-89a0-ea223ad44811.png)

## The weight without regulalirization can be computed using
![image](https://user-images.githubusercontent.com/26680128/32998471-f18611be-cd60-11e7-818c-52ef06d7d816.png)

## The weight with regulalirization can be computed using
![image](https://user-images.githubusercontent.com/26680128/32998495-1d3ec986-cd61-11e7-82fe-55e16408fbba.png)
