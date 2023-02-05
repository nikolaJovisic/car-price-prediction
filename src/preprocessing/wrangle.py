import json
import os
import pandas as pd
import re

from .model_mca import encode_columns
from .utils import days_between

SCRAPING_DAY_STR = "31.1.2023."
SCRAPING_YEAR_INT = int(SCRAPING_DAY_STR.split(".")[-2])


def wrangle():
    """
    Loads and wrangles data.
    :return: Dataframes x,y representing data and labels.
    """
    data = []
    for filename in os.scandir("../../data/textual"):
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
            # date will be used as an equivalent for this
            "ostecenje",  # all data points in one class
            "stanje:",  # just one class left
            "plivajuci zamajac",  # weakly relevant columns or insufficient diversity
            "vlasnistvo",
            "atestiran",
            "datum obnove:",
            "kredit",
            "beskamatni kredit",
            "lizing",
            "gotovinska uplata",
            "broj rata",
            "visina rate",
            "ucesce (depozit)",
            "nacin prodaje",
            "u ponudi od:",
            "boja enterijera",
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
        else max(0, days_between(SCRAPING_DAY_STR, f"{15.}{date}"))
    )
    df.rename(columns={"registrovan do": "registrovan dana"}, inplace=True)

    df["emisiona klasa motora"] = df["emisiona klasa motora"].map(
        lambda euro: int(euro[-1]) if isinstance(euro, str) else 4
    )

    df = encode_columns(df,
                        ["marka", "model", "karoserija", "gorivo", "fiksna cena", "zamena:", "pogon", "menjac", "klima",
                         "materijal enterijera", "poreklo vozila", "zemlja uvoza", "boja"])

    label = df.pop('cena')
    df['cena'] = label

    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    return x, y


def main():
    df = wrangle()
    print(df)


if __name__ == '__main__':
    main()
