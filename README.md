# Car Price Prediction

Model for predicting used car prices, based on structured data, images and textual description.

## Download scraped data

To download data scraped for this project go to the following link: https://drive.google.com/drive/folders/1v3gs5HrOeck9E-tHveOTIMHdygKgmeDn?usp=sharing and place it at the same hierarchical level as src folder.

## Setting scraping environment

In order to scrape data faster add this environment variable to the project (most probably on the path below):
```sh
CRHROME_DATA_PATH = C:\Users\User\AppData\Local\Google\Chrome\User Data
```
This environment variable ensures that the browser is opened with the default user. Without this variable chrome will be open in test automation environment, which is somewhat slower.