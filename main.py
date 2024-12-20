import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

data = pd.read_csv("houseprice.csv",usecols =["SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea",
                                         "Street", "YearBuilt", "LotShape", "1stFlrSF", "2ndFlrSF"]).dropna()

data.shape
data.head()
for col in data.columns:
    print(f"{col} : {len(data[col].unique())}")

import datetime
datetime.datetime.now().year
data['Total Years'] = datetime.datetime.now().year-data['YearBuilt']
data.drop('YearBuilt',axis=1,inplace=True)

cat_cols = ["MSSubClass","MSZoning","Street","LotShape"]
target = "SalePrice"

#kategorik değişkenlere label encoder uygulandı
from sklearn.preprocessing import LabelEncoder
label_encoders={}
for col in cat_cols:
    label_encoders[col]=LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

data.head()

#kateogrik verileri stackleme işlemi (numpy dizisine dönüştü)
cat_cols1 = cat_cols
cat_cols = np.stack([data['MSSubClass'],data['MSZoning'],data['Street'],data['LotShape']],1)
cat_cols

#kategorik değişkenleri numpy dizisinden tensor verisine dönüştürme
cat_cols=torch.tensor(cat_cols,dtype=torch.int64)
cat_cols

#sayısal değişkenlerin seçimi
num_cols = [col for col in data.columns if col not in cat_cols1 and col!= 'SalePrice']
num_cols

#sayısal değişkenleri stackleme ve tensor verisine dönüştürme
num_values = np.stack([data[col].values for col in num_cols],axis=1)
num_values = torch.tensor(num_values,dtype=torch.float)
num_values

#target değişkenin seçimi
target = torch.tensor(data['SalePrice'].values,dtype=torch.float).reshape(-1,1)
target
# kategorik değişkenler için embedding işlemi
cat_dimensions = [len(data[col].unique()) for col in cat_cols1]
cat_dimensions

# Thumbs kuralı
embedding_dimensions = [(x,min(50,(x+1)//2)) for x in cat_dimensions]
embedding_dimensions

#embedding katmanları oluşturuldu (Kategorik değişkenler için)
import torch .nn as nn
import torch.nn.functional as F
embed_representation = nn.ModuleList([nn.Embedding(inp,out) for inp,out in embedding_dimensions])
embed_representation

#embedding vektörleri
embedding_val = []
for i,e in enumerate(embed_representation):
    embedding_val.append(e(cat_cols[:,i]))
embedding_val

z = torch.cat(embedding_val,1)
print(z)
droput = nn.Dropout(.4)
final_embed =droput(z)
print(final_embed)
class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dimensions, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(inp, out) for (inp, out) in embedding_dimensions])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layerlist = []
        n_emb = sum((out for inp, out in embedding_dimensions))
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_num):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_num = self.bn_cont(x_num)
        x = torch.cat([x, x_num], 1)
        x = self.layers(x)
        return x

torch.manual_seed(100)
model = FeedForwardNN(embedding_dimensions,len(num_cols),1,[100,50],p=0.4)
print(model)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

batch_size = 1200
test_size = int(batch_size*0.15)
train_cat_cols = cat_cols[:batch_size-test_size]
test_cat_cols = cat_cols[batch_size-test_size:batch_size]
train_num_cols = num_values[:batch_size-test_size]
test_num_cols = num_values[batch_size-test_size:batch_size]
target_train = target[:batch_size-test_size]
target_test = target[batch_size-test_size:batch_size]

epochs=5000
final_losses = []
for i in range(epochs):
    i=i+1
    y_pred=model(train_cat_cols,train_num_cols)
    loss = torch.sqrt(loss_func(y_pred,target_train))
    final_losses.append(loss)
    if i%10==1:
        print(f"Epoch number: {i} and the loss : {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

final_losses = [loss.detach().numpy() if hasattr(loss, 'detach') else loss for loss in final_losses]
plt.plot(range(epochs), final_losses)
plt.ylabel('RMSE Loss')
plt.xlabel('epoch');

y_pred = ""
with torch.no_grad():
    y_pred = model(test_cat_cols,test_num_cols)
    loss = torch.sqrt(loss_func(y_pred,target_test))
print(f'RMSE: {loss}')

data_verify = pd.DataFrame(target_test.tolist(),columns = ["Test"])
data_predicted = pd.DataFrame(y_pred.tolist(),columns = ["Prediction"])
print(data_predicted)
torch.save(model,'HousePrice.pt')
torch.save(model.state_dict(),'HouseWeights.pt')
embs_size = [(15,8),(5,3),(2,1),(4,2)]
model1 = FeedForwardNN(embs_size,5,1,[100,50],p=0.4)
model1.load_state_dict(torch.load('HouseWeights.pt'))
model1.eval()


