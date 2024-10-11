# Training Artificial neuron lab work


The aim of the task is to train one neuron to solve a two-class problem and to perform an investigation
data sets.

Neurono mokymui yra naudojamas stochastinis gradientinis nusileidimas ir sigmoidinis neuronas.

## Prerequisites:
Installed ucimlrepo.
```
pip install ucimlrepo
``` 
To prepare input data we should: 
- Change `Iris-versicolor` and `Iris-virginica` to 0 and 1 in Iris data.
- Remove id column and all lines containing missing values In breast cancer data.

So it's a nice practice working with Linux tools :D
----
```sh
sed -i 's/Iris-versicolor/0/g; s/Iris-virginica/1/g' ./iris.data
```

```sh
cut -d',' -f2- breast-cancer-wisconsin.data  > temp && mv temp breast-cancer-wisconsin.data;
grep -v '\?' breast-cancer-wisconsin.data > temp && mv temp breast-cancer-wisconsin.data
 ```
---

