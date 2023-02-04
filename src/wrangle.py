import json
import os
import pandas as pd
import re
from utils import days_between

SCRAPING_DATE = "31.1.2023."

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

df = df[
    df["stanje:"] == "polovno vozilo"
]  # very dominant class, only one worth keeping

df.drop(
    columns=[
        "broj oglasa:",  # no information is contained here
        "broj sasije:",  # no information is contained here
        "ostecenje",  # all data points in one class
        "stanje:",  # just one class left
    ],
    inplace=True,
)

df['godiste'] = df['godiste'].map(lambda year: int(year[:-1]))
df['kilometraza'] = df['kilometraza'].map(lambda distance: int(re.sub("[^0-9]", "", distance)))
df['kubikaza'] = df['kubikaza'].map(lambda volume: int(re.sub("[^0-9]", "", volume)))
df['snaga motora'] = df['snaga motora'].map(lambda power: int(power.split('/')[0]))  # only one unit (kw) is needed
df['datum postavke:'] = df['datum postavke:'].map(lambda datum: days_between(datum, SCRAPING_DATE))

for column in df:
    print(column)
    print(df[column].unique())

print(data)
