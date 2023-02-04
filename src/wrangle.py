import json
import os
import pandas as pd
import re
from utils import days_between

SCRAPING_DAY_STR = "31.1.2023."
SCRAPING_MONTH_STR = ".".join(
    SCRAPING_DAY_STR.split(".")[1:],
)
SCRAPING_YEAR_INT = int(SCRAPING_DAY_STR.split(".")[-2])

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
        "datum postavke:",  # all data with known date posted in the last 2 days, so not much info here, scraping
        # date will be used as an equvivalent for this
        "ostecenje",  # all data points in one class
        "stanje:",  # just one class left
    ],
    inplace=True,
)

df["godiste"] = df["godiste"].map(lambda year: SCRAPING_YEAR_INT - int(year[:-1]))
df.rename(columns={"godiste": "starost"}, inplace=True)

df["broj vrata"] = df["broj vrata"].map(lambda doors: int(doors == "4/5 vrata"))
df.rename(columns={"broj vrata": "4/5 vrata"}, inplace=True)

df["strana volana"] = df["strana volana"].map(lambda wheel: int(wheel == "levi volan"))
df.rename(columns={"strana volana": "levi volan"}, inplace=True)

df["kilometraza"] = df["kilometraza"].map(
    lambda distance: int(re.sub("[^0-9]", "", distance))
)

df["kubikaza"] = df["kubikaza"].map(lambda volume: int(re.sub("[^0-9]", "", volume)))

df["snaga motora"] = df["snaga motora"].map(
    lambda power: int(power.split("/")[0])
)  # only one unit (kw) is needed

df["broj sedista"] = df["broj sedista"].map(lambda seats: int(seats.split(" ")[0]))

df["registrovan do"] = df["registrovan do"].map(
    lambda date: 0
    if date == "nije registrovan"
    else days_between(SCRAPING_DAY_STR, f"{15.}{date}")
)
df.rename(columns={"registrovan do": "registrovan dana"}, inplace=True)



for column in df:
    print(column)
    print(df[column].unique())

print(data)
