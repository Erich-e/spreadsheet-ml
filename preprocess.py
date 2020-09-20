import pandas as pd

df = pd.read_csv("iris.csv")
one_hot = pd.get_dummies(df["species"])
df = df.drop("species", axis=1)
df = (df - df.mean()) / df.std()
df = df.join(one_hot)
df = df.sample(frac=1)

num_train = int(len(df) * 0.8)
train = df[:num_train]
test = df[num_train:]
train.to_csv("iris_train.csv", index=False)
test.to_csv("iris_test.csv", index=False)

