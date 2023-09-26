# NBA MVP Prediction 

## Table of Contents
* [Introduction](#introduction)
* [Web Scraping](#web-scraping)
  * [MVP Data](#mvp-data)
  * [Player Data](#player-data)
  * [Team Data](#team-data)
* [Data Cleaning](#data-cleaning)
* [Predictions](#predictions)

## Introduction
The NBA towards the end of every season gives out a variety of awards to players for the impact they've had on the game of basketball that season. The MVP award is one of the most prestigious awards a player can receive for their individual skill. Nowadays, the race for MVP is becoming tighter and tighter as many players are indispensable assets to their teams.

This program aims to predict the MVP using historical data from the 1990 season to the 2023 season. This project required the use of three files, so they will be discussed in chronological order.

## Web Scraping
**To view the full python code for this project, [click here](https://github.com/jidafan/nba-prediction-mvp/blob/main/web_scraping.ipynb).**
### MVP Data
```python
years = list(range(1990,2024))
url_start = "https://www.basketball-reference.com/awards/awards_{}.html"
import requests
for year in years:
    url = url_start.format(year)
    
    data = requests.get(url)
    
    with open("mvps/{}.html".format(year), "w+", encoding="utf-8") as f:
        f.write(data.text)
        time.sleep(15)
```
In this section of code, we are setting the range of MVP data we will be looking at. We download an html file of each MVP season and we include a buffer every 15 seconds, so we don't get timed out from the website.

**Processing the 1990 season**
```python
from bs4 import BeautifulSoup
with open("mvps/1991.html", encoding="utf-8") as f:
    page = f.read(
soup = BeautifulSoup(page, 'html.parser')
mvp_table = soup.find_all(id="mvp")[0]
```
Extracting the MVP table from the downloaded HTML page

**Importing Pandas and setting up MVP dataframe**
```python
import pandas as pd
mvp_1991 = pd.read_html(str(mvp_table))[0]
```
Creating a dataframe for the 1991 season

**Creating an array that holds a dataframe of each MVP season**
```python
dfs = []
for year in years:
    with open("mvps/{}.html".format(year), encoding="utf-8") as f:
        page = f.read()
    soup = BeautifulSoup(page, 'html.parser')
    soup.find('tr', class_="over_header").decompose()
    mvp_table = soup.find(id="mvp")
    mvp = pd.read_html(str(mvp_table))[0]
    mvp["Year"] = year
    dfs.append(mvp)
```
**Converting MVP Data into csv**
```python
mvps = pd.concat(dfs)
mvps.head()
mvps.to_csv("mvps.csv")
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/b4384a2e-9694-4d70-afe3-628297405e12)

### Player Data
```python
player_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"

url = player_stats_url.format(year)
data = requests.get(url)
```
**Importing Selenium**
```python
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys

service = Service(executable_path='chromedriver.exe')
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)
```
Selenium needs to be used as JavaScript is used to load the tables on the website, when trying to scrap from JavaScript-loaded data sometimes the data is missing. So we use Selenium to work around this

**Web Scraping using Selenium**
```python
for year in years:
    url = player_stats_url.format(year)
    driver.get(url)
    driver.execute_script("window.scrollTo(1,10000)")
    time.sleep(10)
    with open("player/{}.html".format(year), "w+", encoding="utf-8") as f:
        f.write(driver.page_source)
```
A for loop is used to scrape through the 1990 season to the 2023 season. We set the code to scroll through the page, so all the data can be loaded.  We have a pause every 10 seconds so we don't get timed out from the site.

**Combining all the Player Data**
```python
dfs = []
for year in years:
    with open("player/{}.html".format(year), encoding="utf-8") as f:
        page = f.read()
    soup = BeautifulSoup(page, 'html.parser')
    soup.find('tr', class_="thead").decompose()
    player_table = soup.find(id="per_game_stats")
    player = pd.read_html(str(player_table))[0]
    player["Year"] = year
    dfs.append(player)
```

**Converting Player Data into CSV**
```python
players = pd.concat(dfs)
players.to_csv("players.csv")
```

### Team Data
```python
team_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_standings.html"
for year in years:
    url = team_stats_url.format(year)
    
    data = requests.get(url)
    
    with open("team/{}.html".format(year), "w+", encoding="utf-8") as f:
        f.write(data.text)
        time.sleep(15)
```
Looking at the team stats from the 1990 season to the 2023 season, and we set a buffer every 15 seconds after we download the HTML page

**Combining all the Team Data**
```python
dfs = []
for year in years:
    with open("team/{}.html".format(year), encoding="utf-8") as f:
        page = f.read()
    soup = BeautifulSoup(page, 'html.parser')
    soup.find('tr', class_="thead").decompose()
    team_table = soup.find(id="divs_standings_E")
    team = pd.read_html(str(team_table))[0]
    team["Year"] = year
    team["Team"] = team["Eastern Conference"]
    del team["Eastern Conference"]
    dfs.append(team)
    
    soup = BeautifulSoup(page, 'html.parser')
    soup.find('tr', class_="thead").decompose()
    team_table = soup.find(id="divs_standings_W")
    team = pd.read_html(str(team_table))[0]
    team["Year"] = year
    team["Team"] = team["Western Conference"]
    del team["Western Conference"]
    dfs.append(team)
```
Set up the team data so it is distinguished from the Western Conference and Eastern Conference

**Converting Team Data into CSV**
```python
teams = pd.concat(dfs)
teams
teams.to_csv("teams.csv")
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/66da760d-4d77-4499-b885-93ff3ebb5216)

## Data Cleaning
**To view the full python code for this project, [click here](https://github.com/jidafan/nba-prediction-mvp/blob/main/data_cleaning.ipynb).**
**Loading in the MVP Data**
```python
import pandas as pd
mvps = pd.read_csv("mvps.csv")
mvps
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/85d9809e-cbe7-4c63-ace9-2679f6db3b14)

**Overwriting MVP dataframe**
```python
mvps = mvps[["Player", "Year", "Pts Won", "Pts Max", "Share"]]
mvps
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/db298c9f-ed87-4964-8cf3-7d422333e4bc)

**Loading in the Player Data**
```python
players = pd.read_csv("players.csv")
players
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/db1348ad-d22a-426f-aaf8-e84da846c510)

**Deleting unwanted columns**
```python
del players["Unnamed: 0"]
del players["Rk"]
players
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/7c1a6982-a6b1-4907-8065-6c88c5d29f61)

**Cleaning Player Data**
```python
players["Player"] = players["Player"].str.replace("*","", regex = False)
```
Removing * from Player Data dataframe

**Grouping the players in dataframe**
```python
def single_row(df):
    if df.shape[0] == 1:
        return df
    else:
        row = df[df["Tm"] == "TOT"]
        row["Tm"] = df.iloc[-1,:]["Tm"]
        return row

players = players.groupby(["Player","Year"]).apply(single_row)

players.head(20)
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/d1cf68a0-44b3-42ed-b92c-491faaa5a9d2)

**Dropping Level of dataframe**
```python
players.index = players.index.droplevel()
players.index = players.index.droplevel()
players
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/b707ff3e-9924-4576-9a81-745d696aee23)

**Combined the Player Data and MVP Data**
```python
combined = players.merge(mvps, how="outer", on=["Player","Year"])
combined.head(5)
combined[combined["Pts Won"] > 0]
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/365091ef-b09e-4117-925e-f557b1ae59cc)
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/86349cf3-a8ae-4196-b051-584911b53f11)

**Filling in the NaN values with 0**
```python
combined[["Pts Won", "Pts Max", "Share"]] = combined[["Pts Won", "Pts Max", "Share"]].fillna(0)
combined
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/b98fc109-6e1e-4d3b-9dec-497878f46d75)

**Loading in Team Data**
```python
teams = pd.read_csv("teams.csv")
teams
teams.head(30)
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/a472148c-6c0c-4d72-80e2-54e736281441)

**Removing Division Row**
```python
teams = teams[~teams["W"].str.contains("Division")]
teams
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/b02f7fc3-2025-4477-b901-6fa4ee82f80f)

**Cleaning Team Data**
```python
teams = teams[~teams["W"].str.contains("Division")]
teams
teams["Team"] = teams["Team"].str.replace("*","", regex = False)
teams
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/d01ac7fe-9adc-4543-8b3b-ae721254c8f6)

**Adjusting Team Name in dataframes**
```python
teams["Team"].unique()
combined["Tm"].unique()
nicknames = {}

with open("nicknames.csv") as f:
    lines = f.readlines()
    for line in lines[1:]:
        abbrev,name = line.replace("\n","").split(",")
        nicknames[abbrev] = name
nicknames
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/1681b0f6-e49b-4c2d-8388-134077a077fd)


```python
combined["Team"] = combined["Tm"].map(nicknames)
combined.head(5)
```

![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/f5723aa1-c0cd-4ebb-88ae-cf157887b6e9)

In this section of code, we are making sure that the method of looking up the team name in the data frame is consistent before we merge them together. In the teams dataframe, they are referred through the Long Form (i.e., Golden State Warriors),
but in the combined dataframe, it is through the short form (i.e., GSW)

**Merging the dataframes**
```python
stats = combined.merge(teams, how="outer", on=["Team","Year"])
stats
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/2269a147-e09f-4c11-8ff4-7408e6a35532)

**Converting the datatypes of the dataframe columns**
```python
stats = stats.apply(pd.to_numeric, errors="ignore")
stats.dtypes
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/8800b2e8-2cd9-42a2-8f4b-49ac8456373a)

**Fixing the GB column**
```python
stats["GB"].unique()
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/7231952b-fd37-4337-bdd0-4b0a867954b7)

```python
stats["GB"] = stats["GB"].str.replace("—","0")
stats["GB"].unique()
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/ea1437ec-c57e-4f6a-8c6a-6579d65af32c)

```python
stats["GB"] = pd.to_numeric(stats["GB"])
```

**Writing the combined dataframes into a CSV**
```python
stats.to_csv("player_mvp_stats.csv")
```

**Looking at the correlation of each stat to the share column**
```python
stats.corr(numeric_only = True)["Share"]
stats.corr(numeric_only = True)["Share"].plot.bar()
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/1bad766e-a7fe-41cd-95b3-2c55cc6d2743)

## Predictions
**To view the full python code for this project, [click here](https://github.com/jidafan/nba-prediction-mvp/blob/main/prediction.ipynb).**

**Reading in stats csv**
```python
import pandas as pd
stats = pd.read_csv("player_mvp_stats.csv")
stats
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/b20e471e-bc4d-4b3d-b79d-c59e66f4c910)

**Looking at the null values in the dataframe
```python
pd.isnull(stats).sum()
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/d41b7070-0393-4e96-8a6c-f21707f5146b)

Shows the sum of all the null values in the dataframe

**Looking at players who have shot 0 3 Pointers**
```python
stats[pd.isnull(stats["3P%"])][["Player","3PA"]]
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/4b76873b-1491-455b-bcb2-0abc569d4dd9)

The reason why there are so many null values in stats dataframe is that there are a lot of players who have never shot a 3. Thus, there will be a null value for their 3P%.

**Looking at players who have shot 0 freethrows**
```python
stats[pd.isnull(stats["FT%"])][["Player","FTA"]]
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/35bb13e6-691a-436d-b38c-62a6987ae8dd)

The reason why there are so many null values in stats dataframe is that there are a lot of players who have never shot a freethrow. Thus, there will be a null value for their FT%.

```python
stats = stats.fillna(0)
```
To fix this, we set all the null values to 0

**Creating predictors column**
```python
predictors = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
       '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year', 'W', 'L', 'W/L%', 'GB', 'PS/G',
       'PA/G', 'SRS']
```

**Creating Training and Test set**
```python
train = stats[~(stats["Year"] == 2023)]
test = stats[stats["Year"] == 2023]
```

**Importing our model and creating predictions**
```python
from sklearn.linear_model import Ridge

reg = Ridge(alpha=0.1)
reg.fit(train[predictors], train["Share"])

predictions = reg.predict(test[predictors])
predictions = pd.DataFrame(predictions, columns = ["predictions"], index=test.index)
predictions
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/97c83719-6cd8-4ba4-a16b-645f4c64129b)

We are predicting the player's share of MVP votes using ridge regression

**Combining predictions with test set**
```python
combination = pd.concat([test[["Player", "Share"]], predictions], axis = 1)
combination
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/000b997e-318c-443a-96e9-af7b2332dd66)

**Importing our error metric**
```python
from sklearn.metrics import mean_squared_error

mean_squared_error(combination["Share"], combination["predictions"])
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/1ff34ed1-d8f3-4fa5-b48f-77554ca99231)

**Looking at the share values**
```python
combination["Share"].value_counts()
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/8fae911f-d6a7-458a-bf49-4a35d3d7d071)

**Inputting actual MVP rank**
```python
combination = combination.sort_values("Share", ascending = False)
combination["Rk"] = list(range(1,combination.shape[0]+1))
combination.head(10)
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/87fb856c-11b1-4511-87a9-955775155df0)

**Looking at predicted ranks**
```python
combination = combination.sort_values("predictions", ascending = False)
combination["Predicted_Rk"] = list(range(1,combination.shape[0]+1))
combination.head(10)
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/2ad72781-ac98-4d3a-91da-09a1e98c3a2a)

```python
combination.sort_values("Share", ascending = False).head(10)
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/6895124e-467f-4841-aced-929ca4d8bf22)

**Creating new error metric**
```python
def find_ap(combination):
    actual = combination.sort_values("Share", ascending = False).head(5)
    predicted = combination.sort_values("predictions", ascending = False)
    ps = []
    found = 0 
    seen = 1
    for index, row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            ps.append(found/seen)
        seen += 1
    return sum(ps)/len(ps)

find_ap(combination)

aps = []
all_predictions = []
for year in years[5:]:
    train = stats[stats["Year"] < year]
    test = stats[stats["Year"] == year]
    reg.fit(train[predictors], train["Share"])
    predictions = reg.predict(test[predictors])
    predictions = pd.DataFrame(predictions, columns = ["predictions"], index=test.index)
    combination = pd.concat([test[["Player", "Share"]], predictions], axis = 1)
    all_predictions.append(combination)
    aps.append(find_ap(combination))

sum(aps)/len(aps)
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/917c5ec6-6586-4214-9ffb-a63d56cdd583)

Creating a new error metric called accuracy precision, which rates us on how accurately we can determine the players in the top 5 for MVP share votes

**Adding difference in dataframe**
```python
def add_ranks(combination):
    combination = combination.sort_values("Share", ascending = False)
    combination["Rk"] = list(range(1,combination.shape[0]+1))
    combination = combination.sort_values("predictions", ascending = False)
    combination["Predicted_Rk"] = list(range(1,combination.shape[0]+1))
    combination["Diff"] = combination["Rk"] - combination["Predicted_Rk"]
    return combination

ranking = add_ranks(all_predictions[1])
ranking[ranking["Rk"]<= 5].sort_values("Diff", ascending = False)
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/cb0c8c36-0ce5-4f66-99a0-fa39dd5373d3)

Shows how off we are from predicting the top 5 players in MVP voting

**Creating backtest function**
```python
def backtest(stats, model, year, predictors):
    aps = []
    all_predictions = []
    for year in years[5:]:
        train = stats[stats["Year"] < year]
        test = stats[stats["Year"] == year]
        model.fit(train[predictors], train["Share"])
        predictions = model.predict(test[predictors])
        predictions = pd.DataFrame(predictions, columns = ["predictions"], index=test.index)
        combination = pd.concat([test[["Player", "Share"]], predictions], axis = 1)
        combination = add_ranks(combination)
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    return sum(aps)/len(aps), aps, pd.concat(all_predictions)

    mean_ap, aps, all_predictions = backtest(stats, reg, years, predictors)

    mean_ap
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/852e7631-d1f9-4204-af7a-f84d5e6d729b)

Creating a backtest function to try to improve our accuracy

**Looking at worst differences**
```python
all_predictions[all_predictions["Rk"] <= 5].sort_values("Diff").head(10)
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/58dacf31-3cb5-4c85-ac91-f4dfb9baf4e6)

**Looking at regression coefficient**
```python
pd.concat([pd.Series(reg.coef_), pd.Series(predictors)], axis=1).sort_values(0, ascending=False)
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/ede9e3dc-d190-4021-a666-97bc8f37662d)

Looks at which variables the regression model think contribute highest to MVP voting

**Looking at player stats and average stats for season**
```python
stat_ratios = stats[["PTS", "AST", "STL", "BLK", "3P", "Year"]].groupby("Year").apply(lambda x: x/x.mean()).reset_index(drop=True)
stat_ratios
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/8c1d630e-4994-4238-88a3-9ac55603dfa8)

**Add to stats dataframe**
```python
stats = stats.reset_index(drop=True)
stats
stats[["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]] = stat_ratios[["PTS", "AST", "STL", "BLK", "3P"]]
stats.head()
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/c3873515-1758-4ad3-8f0f-52126a4510e6)

**Adding new predictors**
```python
predictors += ["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]
mean_ap, aps, all_predictions = backtest(stats, reg, years, predictors)
mean_ap
```

![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/e8b25f5f-e4b0-4b1b-86fe-0cc4b4666e9f)

Adding our new predictors and testing our model, the ap score has increased which means our accuracy has improved

**Looking at positions**
```python
stats["NPos"] = stats["Pos"].astype("category").cat.codes
stats["NTm"] = stats["Tm"].astype("category").cat.codes
stats["Pos"].unique()
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/c8c9ab26-8f9e-46fa-97ed-bf942536fcfb)

**Looking at Team Wins**
```python
stats["NTm"].value_counts()
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/ab9cce67-7ca9-4ee2-a90f-4ba31ed80904)

**Implementing RandomForest**
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 50, random_state = 1, min_samples_split = 5)

mean_ap, aps, all_predictions = backtest(stats, rf, years[23:], predictors)

mean_ap
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/a2d1a829-e3aa-4404-8e6b-8cc4528e21e5)'

RandomForest improved our accuracy

**Using more years as training for backtest**
```python
mean_ap, aps, all_predictions = backtest(stats, reg, years[23:], predictors)
mean_ap
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/84736c83-86c2-4ba8-9c8a-92fd3eb9887f)

Decreased our accuracy

**Implementing Gradient Boosting**
```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=8, random_state=0, loss='squared_error')

mean_ap, aps, all_predictions = backtest(stats, gbr, years, predictors)

mean_ap
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/29f8827d-81ae-473f-aeb4-91544b73d19e)

Decreased our accuracy by a lot

**Implementing Elastic Net**
```python
from sklearn.linear_model import ElasticNetCV
import numpy as np

encv = ElasticNetCV(cv=15, random_state=1, l1_ratio=[0.1,0.5,0.7], max_iter=2000, n_alphas=500, selection = "random")

mean_ap, aps, all_predictions = backtest(stats, encv, years, predictors)

mean_ap
```
![image](https://github.com/jidafan/nba-prediction-mvp/assets/141703009/56235b75-d17a-40e7-95a4-cc024f6e03a1)

Improved our accuracy by a lot

**Implementing LightGBM**
```python
import lightgbm as lgb

hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1','l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    "max_depth": 6,
    "num_leaves": 64,  
    "max_bin": 128,
    "num_iterations": 10000
}

lgbm = lgb.LGBMRegressor(**hyper_params)

mean_ap, aps, all_predictions = backtest(stats, lgbm, years, predictors)
mean_ap
```

Didn't improve our accuracy
