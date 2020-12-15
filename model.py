import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv('data2.csv')

df1 = df.drop(['Patid'],axis=1)

X = df1.iloc[:,:-17].values
y = df1.iloc[:,3600:3617].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=3600, out_features=3000)
        self.fc2 = nn.Linear(in_features=3000, out_features=2500)
        self.fc3 = nn.Linear(in_features=2500, out_features=2000)
        self.fc4 = nn.Linear(in_features=2000, out_features=1500)
        self.fc5 = nn.Linear(in_features=1500, out_features=1000)
        self.fc6 = nn.Linear(in_features=1000, out_features=500)
        self.fc7 = nn.Linear(in_features=500, out_features=400)
        self.fc8 = nn.Linear(in_features=400, out_features=300)
        self.fc9 = nn.Linear(in_features=300, out_features=150)
        self.fc10 = nn.Linear(in_features=150, out_features=40)
        self.output = nn.Linear(in_features=40, out_features=17)
        self.dropout = nn.Dropout(0.5)
 
    def forward(self, x):
        x = F.softmax(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x))
        x = self.dropout(x)
        x = F.softmax(self.fc3(x))
        x = self.dropout(x)
        x = F.softmax(self.fc4(x))
        x = self.dropout(x)
        x = F.softmax(self.fc5(x))
        x = self.dropout(x)
        x = F.softmax(self.fc6(x))
        x = self.dropout(x)
        x = F.softmax(self.fc7(x))
        x = self.dropout(x)
        x = F.softmax(self.fc8(x))
        x = self.dropout(x)
        x = F.softmax(self.fc9(x))
        x = self.dropout(x)
        x = F.softmax(self.fc10(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

model = ANN()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.01)
loss_arr = []

epochs = 20
i=0
for e in range(epochs):
    running_loss = 0
    for images, labels in zip(X_train,y_train):
        # Flatten MNIST images into a 784 long vector
        #images = images.view(images.shape[0], -1)
        i+=1
        # TODO: Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = torch.mean((output - labels)**2)
        #output = output.reshape([-1,1])
        #labels = labels.reshape([17,1])
        #loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i%10 == 0:
            print('Epoch:', i ,'Loss:',loss.item())

# getting the threshold
s=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
c=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(200):
    for j in range(17):
        if(y_test[i][j]==1 and (model.forward(X_test[i])[j])>0):
            s[j]+=model.forward(X_test[i])[j]
            c[j]+=1
# average of threshold
a=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(17):
    a[i]=s[i]/c[i]
    
y_pred = torch.zeros(200,17)
for i in range(200):
    for j in range(17):
        if((model.forward(X_test[i])[j])>=a[j]):
            y_pred[i][j]=1
            
y_pred_copy = pd.DataFrame(y_pred)
y_test_copy = pd.DataFrame(y_test)

pos=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(200):
    for j in range(17):
        if(y_pred[i][j]==y_test[i][j]):
            pos[j]+=1

for i in range(17):
    pos[i]=pos[i]/200


acc=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(17):
    acc[i]=pos[i]/c[i]
    
pos=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(200):
    for j in range(17):
        if(y_pred[i][j]==1 and y_test[i][j]==1):
            pos[j]+=1
