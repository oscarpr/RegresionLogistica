# -*- coding: utf-8 -*-
from sklearn.linear_model.logistic   import LogisticRegression
from sklearn                            import metrics
import numpy as np

matrix = np.genfromtxt('data.txt', delimiter = ',')

m=0
for m in range(297):
    if matrix[m][13] != 0:
        matrix[m][13] = 1 

#Separamos la matriz de entrenamiento, de validacion y prueba. con sus respectivas y's
x_train = np.zeros((237,13))
x_training = np.zeros((178,13))
x_validation = np.zeros((59,13))
y_train = np.zeros((237,1))
x_test = np.zeros((60,13))
y_training = np.zeros((178,1))
y_validation = np.zeros((59,1))
y_test = np.zeros((60,1))
i=0
j=0
ii=-1
iii=-1
for i in range(297):
    if (i>177) and (i<237):
        ii += 1
    elif i>238:
        iii += 1
    for j in range(13):
        if (i<178):
            x_train[i][j] = matrix[i][j]
            y_train[i] = matrix[i][13]
            x_training[i][j] = matrix[i][j]
            y_training[i] = matrix[i][13]
        elif (i>177) and (i<237):
            x_train[i][j] = matrix[i][j]
            y_train[i] = matrix[i][13]
            x_validation[ii][j] = matrix[i][j]
            y_validation[ii] = matrix[i][13]
        elif i>238:
            x_test[iii][j] = matrix[i][j]
            y_test[iii] = matrix[i][13]


#Implementacion del algoritmo de regresion logistica___________________________________________________________________
logisticReg = LogisticRegression(penalty= 'l2', C = 2.0)
logisticReg.fit(x_training, y_training)

#Usamos los datos de validacion
validations = logisticReg.predict(x_validation)

for k in range(59):
    if validations[k] > 1:
        validations[k] = 1

print('Valores de entrenamiento')
print(metrics.classification_report(y_validation, logisticReg.predict(x_validation)))


logisticReg.fit(x_train, y_train) #Añadimos los datos de validacion antes de probar

#Usamos los datos de prueba
tests = logisticReg.predict(x_test)
for k in range(59):
    if tests[k] > 1:
        tests[k] = 1
                
print('Valores de prueba')
print(metrics.classification_report(y_test, logisticReg.predict(x_test)))