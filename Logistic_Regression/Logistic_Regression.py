import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

#### Processing dataset
def preprocess_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    return processed_df


#### Reading Data Set using pandas
df = pd.read_csv("Dataset/train.csv")
df = df.drop(['Name','Ticket','Cabin','PassengerId'], axis=1)
df = df.dropna()

#print(df)
processed_df = preprocess_df(df)
input_NN = processed_df.values[:,1:]
output_NN = processed_df.values[:,0]

#### Building the Neural Network
model = Sequential()
model.add(Dense(10,input_dim = input_NN.shape[1], activation= 'relu'))
#model.add(Dense(4,input_dim = 10, activation= 'relu'))
model.add(Dense(1,activation= 'sigmoid'))

#### Compile models
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#### Train the model
model.fit(input_NN,output_NN, epochs=1000, batch_size=16)

#### Evaluate the model
scores = model.evaluate(input_NN,output_NN)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#### Importing test dataset
df = pd.read_csv("Dataset/test.csv")
#df1 = pd.read_csv("Dataset/gender_submission.csv")
#df = pd.merge(df1, df)
df = df.drop(['Name','Ticket','Cabin'], axis=1)
#df = df.dropna()
df1 = pd.DataFrame()
df1['PassengerId'] = df['PassengerId'].copy()
df = df.drop(['PassengerId'], axis=1)
print(df)

processed_df = preprocess_df(df)
test_input = processed_df.values
#actual_output = processed_df.values[:,1]

predicted_output = np.round(model.predict(test_input)).astype(int)
df1['Survived'] = predicted_output
df1['Survived'] = df1['Survived'].map(lambda s: 1 if s >= 0.5 else 0)
print(df1)
df1[['PassengerId','Survived']].to_csv('./Dataset/My_Results.csv', index=False)
#df_output = pd.DataFrame()
