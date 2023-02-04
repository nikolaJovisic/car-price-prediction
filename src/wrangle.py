import json
import os
import pandas as pd


data = []
for filename in os.scandir("../data/textual"):
    with open(filename) as file:
        datum = json.load(file)
        data.append(
            datum["opste informacije"]
            | datum["dodatne informacije"]
            | {"sigurnost": len(datum["sigurnost"])}
            | {"oprema": len(datum["oprema"])}
            | {"cena": datum["cena"]}
        )
df = pd.DataFrame(data)

for column in df:
    print(column)
    print(df[column].unique())

print(data)
