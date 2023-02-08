import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd

vectorZ = sys.argv[1]

df = pd.read_csv('src/server/csv-data/data.csv', sep=';', on_bad_lines='skip')
df["embedding"] = df.embedding.apply(eval).apply(np.array)

# print(df["embedding"])
# arrayOfArraysData = vectorX[:-1]
# arrayOfArraysData = arrayOfArraysData.split('$,')
# arrayOfArraysTarget = vectorY.split(',')

# for i in range(len(arrayOfArraysData)):
#    arrayOfArraysData[i] = arrayOfArraysData[i].split(',')
#    arrayOfArraysData[i] = [float(x) for x in arrayOfArraysData[i]]

# arrayOfArraysData = np.array(arrayOfArraysData)
# arrayOfArraysTarget = np.array(arrayOfArraysTarget)
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.classes, test_size=0.3, random_state=43
)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
report = classification_report(y_test, preds)
arrayOfPredict = vectorZ.split(',')
arrayOfPredict = [float(x) for x in arrayOfPredict]
arrayOfPredict = [np.array(arrayOfPredict)]
print(report)
clf.fit(X_train, y_train)

result = clf.predict(arrayOfPredict)
print(result[0])
sys.stdout.flush()
