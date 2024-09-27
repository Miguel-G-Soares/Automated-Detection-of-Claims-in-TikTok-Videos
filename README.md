# Automated Detection of Claims in TikTok Content

## Project Overview

This project aims to develop a predictive model that classifies whether TikTok videos contain user claims or opinions. The goal is to help TikTok's moderation team streamline the process of reviewing content that has been flagged by prioritizing user reports more effectively. Using a dataset of TikTok content, machine learning models were trained, evaluated and tested on their ability to accurately classify content as either claim or opinion-based.


## Business Problem

TikTok relies on user reports to identify potentially problematic content (misinformation, claims, inappropriate material, etc...). With millions of users creating a vast amount of data everyday, Tiktok has a significant challenge in processing all of the reports made.

The content moderation team needs to assess whether flagged content contains factual claims or simply expresses opinions. Classifying content into these two categories can help TikTok reduce the workload on moderators, allowing them to prioritize claim-based content that may require quicker and/or more thorough reviews.This will not only reduce response time but also improve the user experience by handling harmful or misleading content more efficiently.


## Data

The dataset collected includes several variables that can help identify claim/oppinion based content:

* Video id: Identifying number assigned to video upon publication on TikTok
* Video duration: How long the published video is in seconds
* Video transcription text: Transcribed text of the words spoken in the video
* Verified status: Whether the TikTok user who published the video is either “verified” or “not verified”
* Author ban status: Whether the TikTok user who published the video is either “active”, “under scrutiny” or “banned”
* Video view count: The total number of views a video has
* Video like count: The total number of likes a video has
* Video share count: The total number of times a video has been shared
* Video download count: The total number of times a video has been downloaded
* Video comment count: The total number of comments a video has
* Claim status (Target): Whether the video has a "claim" or an "opinion".

## Modelling


```python
# Data manipulation
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Data modeling
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Data evaluation
import sklearn.metrics as metrics
from xgboost import plot_importance
```


```python
data = pd.read_csv("tiktok_dataset.csv")
```

### EDA


```python
data.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (19382, 12)




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19382 entries, 0 to 19381
    Data columns (total 12 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   #                         19382 non-null  int64  
     1   claim_status              19084 non-null  object 
     2   video_id                  19382 non-null  int64  
     3   video_duration_sec        19382 non-null  int64  
     4   video_transcription_text  19084 non-null  object 
     5   verified_status           19382 non-null  object 
     6   author_ban_status         19382 non-null  object 
     7   video_view_count          19084 non-null  float64
     8   video_like_count          19084 non-null  float64
     9   video_share_count         19084 non-null  float64
     10  video_download_count      19084 non-null  float64
     11  video_comment_count       19084 non-null  float64
    dtypes: float64(5), int64(3), object(4)
    memory usage: 1.8+ MB
    


```python
data.describe()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19382.000000</td>
      <td>1.938200e+04</td>
      <td>19382.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9691.500000</td>
      <td>5.627454e+09</td>
      <td>32.421732</td>
      <td>254708.558688</td>
      <td>84304.636030</td>
      <td>16735.248323</td>
      <td>1049.429627</td>
      <td>349.312146</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5595.245794</td>
      <td>2.536440e+09</td>
      <td>16.229967</td>
      <td>322893.280814</td>
      <td>133420.546814</td>
      <td>32036.174350</td>
      <td>2004.299894</td>
      <td>799.638865</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.234959e+09</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4846.250000</td>
      <td>3.430417e+09</td>
      <td>18.000000</td>
      <td>4942.500000</td>
      <td>810.750000</td>
      <td>115.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9691.500000</td>
      <td>5.618664e+09</td>
      <td>32.000000</td>
      <td>9954.500000</td>
      <td>3403.500000</td>
      <td>717.000000</td>
      <td>46.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14536.750000</td>
      <td>7.843960e+09</td>
      <td>47.000000</td>
      <td>504327.000000</td>
      <td>125020.000000</td>
      <td>18222.000000</td>
      <td>1156.250000</td>
      <td>292.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19382.000000</td>
      <td>9.999873e+09</td>
      <td>60.000000</td>
      <td>999817.000000</td>
      <td>657830.000000</td>
      <td>256130.000000</td>
      <td>14994.000000</td>
      <td>9599.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Checking for missing values


```python
data.isna().sum()
```




    #                             0
    claim_status                298
    video_id                      0
    video_duration_sec            0
    video_transcription_text    298
    verified_status               0
    author_ban_status             0
    video_view_count            298
    video_like_count            298
    video_share_count           298
    video_download_count        298
    video_comment_count         298
    dtype: int64



There are at least 298 rows with missing data


```python
data[data.isna().any(axis=1)].head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19084</th>
      <td>19085</td>
      <td>NaN</td>
      <td>4380513697</td>
      <td>39</td>
      <td>NaN</td>
      <td>not verified</td>
      <td>active</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19085</th>
      <td>19086</td>
      <td>NaN</td>
      <td>8352130892</td>
      <td>60</td>
      <td>NaN</td>
      <td>not verified</td>
      <td>active</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19086</th>
      <td>19087</td>
      <td>NaN</td>
      <td>4443076562</td>
      <td>25</td>
      <td>NaN</td>
      <td>not verified</td>
      <td>active</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19087</th>
      <td>19088</td>
      <td>NaN</td>
      <td>8328300333</td>
      <td>7</td>
      <td>NaN</td>
      <td>not verified</td>
      <td>active</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19088</th>
      <td>19089</td>
      <td>NaN</td>
      <td>3968729520</td>
      <td>8</td>
      <td>NaN</td>
      <td>not verified</td>
      <td>active</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



It appears that all observations with missing data only have information on the video's id, duration and author.

Observtions are missing the most important information and as such should be removed


```python
data.dropna(inplace=True)
```

#### Checking for duplicates


```python
data.duplicated().sum()
```




    0



There isn't any duplicated data

#### Deleting purposeless information


```python
data.drop(columns=['#','video_id'],inplace=True)
```

#### Checking for outliers


```python
for column in data.select_dtypes(include=['int64','float64']).columns:
    plt.figure(figsize=(5,1))
    sns.boxplot(data[column],orient='h')
    plt.xlabel(column)
    plt.show()
```


    
![png](images/output_22_0.png)
    



    
![png](images/output_22_1.png)
    



    
![png](images/output_22_2.png)
    



    
![png](images/output_22_3.png)
    



    
![png](images/output_22_4.png)
    



    
![png](images/output_22_5.png)
    


Video like, share comment and download counts are significantly right-skewed, with most videos having low popularity scores while the rare few that go viral increase tremendously

Checking predictor feature class balance


```python
data.claim_status.value_counts(normalize=True)
```




    claim_status
    claim      0.503458
    opinion    0.496542
    Name: proportion, dtype: float64



#### Feature transformation


```python
data.claim_status.unique()
```




    array(['claim', 'opinion'], dtype=object)




```python
data.verified_status.unique()
```




    array(['not verified', 'verified'], dtype=object)




```python
data.author_ban_status.unique()
```




    array(['under review', 'active', 'banned'], dtype=object)



The three categorical features to handle are claim_status, verified_status, and author_ban_status:

Two of them are binary
* claim_status: opinion , claim
* verified_status: not verified , verified

While the last is ordinal in nature
* author_ban_status: banned , under review , active


```python
# One-hot encoding the binary categorical features
data.claim_status = data.claim_status.replace({'opinion': 0, 'claim': 1})
data.verified_status = data.verified_status.replace({'not verified': 0, 'verified': 1})

# Label encoding the ordinal feature
data.author_ban_status = data.author_ban_status.replace({'banned':0,'under review': 1, 'active': 2})
```


```python
data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_status</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>text_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>0</td>
      <td>1</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>97</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>0</td>
      <td>2</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
      <td>107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>0</td>
      <td>2</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
      <td>137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>0</td>
      <td>2</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
      <td>131</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>0</td>
      <td>2</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
      <td>128</td>
    </tr>
  </tbody>
</table>
</div>



#### Feature extraction


```python
data['text_len']=data.video_transcription_text.apply(len)
```

Text length of videos with claims versus opinions


```python
data[['claim_status','text_len']].groupby(['claim_status']).mean().reset_index()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_status</th>
      <th>text_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>claim</td>
      <td>95.376978</td>
    </tr>
    <tr>
      <th>1</th>
      <td>opinion</td>
      <td>82.722562</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.histplot(x=data.text_len,hue=data.claim_status)
plt.show()
```


    
![png](images/output_36_0.png)
    


Tokenizing the text column


```python
# Tokenizing text into 2-grams
c_vec = CountVectorizer(ngram_range=(2, 3),
                            max_features=15,
                            stop_words='english')
c_vec
```




```python
# Fitting vectorizer into the data
count_data = c_vec.fit_transform(data['video_transcription_text']).toarray()
count_data
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int64)




```python
# Creating the dataframe counting the 15 most used n-grams
count_df = pd.DataFrame(data=count_data, columns=c_vec.get_feature_names_out())

count_df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>colleague discovered</th>
      <th>colleague learned</th>
      <th>colleague read</th>
      <th>discussion board</th>
      <th>friend learned</th>
      <th>friend read</th>
      <th>internet forum</th>
      <th>learned media</th>
      <th>learned news</th>
      <th>media claim</th>
      <th>news claim</th>
      <th>point view</th>
      <th>read media</th>
      <th>social media</th>
      <th>willing wager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19079</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19080</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19081</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19082</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19083</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>19084 rows × 15 columns</p>
</div>




```python
# Updating the original datarame
data = pd.concat([data.drop(columns='video_transcription_text'),count_df],axis=1)

data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_status</th>
      <th>video_duration_sec</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>text_len</th>
      <th>...</th>
      <th>friend read</th>
      <th>internet forum</th>
      <th>learned media</th>
      <th>learned news</th>
      <th>media claim</th>
      <th>news claim</th>
      <th>point view</th>
      <th>read media</th>
      <th>social media</th>
      <th>willing wager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>59</td>
      <td>0</td>
      <td>1</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>97</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>32</td>
      <td>0</td>
      <td>2</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
      <td>107</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>31</td>
      <td>0</td>
      <td>2</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
      <td>137</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>25</td>
      <td>0</td>
      <td>2</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
      <td>131</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>19</td>
      <td>0</td>
      <td>2</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
      <td>128</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



## Creating the ML models


```python
x=data.drop(columns='claim_status')
y=data.claim_status
```


```python
#Creating 3 different subsets with the following percentages train-0.6 , validate-0.2 , test-0.2

x_tr,x_test,y_tr,y_test = train_test_split(x,y,random_state=0,test_size=0.2)
x_train,x_val,y_train,y_val=train_test_split(x_tr,y_tr,random_state=0,test_size=0.25)
```


```python
# Making sure all dataframes are as expected
for type in ['train','test','val']:
    print('x'+type+':',globals()['x_'+type].shape)
    print('y'+type+':',globals()['y_'+type].shape)
```

    xtrain: (11450, 24)
    ytrain: (11450,)
    xtest: (3817, 24)
    ytest: (3817,)
    xval: (3817, 24)
    yval: (3817,)
    

### Random Forest

The evaluation metric chosen is recall as it most emphasizes a reduction in False Negatives. The Tiktok team wants to be as sure as possible that videos flagged by the model are indeed claims or not


```python
# Instantiating the random forest model
rf=RandomForestClassifier(random_state=0)

# Selecting hyperparameters to tune
cv_params = {
    'max_depth': [5, 7, None],
    'max_features': [0.3, 0.6],
    'max_samples': [0.7],
    'min_samples_leaf': [1,2],
    'min_samples_split': [2,3],
    'n_estimators': [75,100,200],
}

# Selecting scores
scoring=('precision','accuracy','recall','f1')

# Instantiating the Cross-Validation classifier
clf=GridSearchCV(rf,param_grid=cv_params,scoring=scoring,cv=5,refit='recall')
```


```python
%%time
clf.fit(x_train,y_train)
```

    CPU times: total: 9min 28s
    Wall time: 9min 49s
    



```python
print(clf.best_params_)
print(clf.best_score_)
```

    {'max_depth': None, 'max_features': 0.6, 'max_samples': 0.7, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 75}
    0.9948228253467271
    

### Gradient Boosting


```python
xgb = XGBClassifier( objective = 'binary:logistic' , random_state = 0 )

xgb_params ={'max_depth': [4,8,12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300, 500]
             }

clf_xgb = GridSearchCV(xgb,param_grid=xgb_params,scoring=scoring,cv=5,refit='recall')
```


```python
%%time
clf_xgb.fit(x_train, y_train)
```

    CPU times: total: 2min 6s
    Wall time: 39.6 s
    



```python
print(clf_xgb.best_params_)
print(clf_xgb.best_score_)
```

    {'learning_rate': 0.1, 'max_depth': 12, 'min_child_weight': 3, 'n_estimators': 300}
    0.989645054622456
    

### Evaluating the models

Random Forest:


```python
y_rf_pred=clf.best_estimator_.predict(x_val)
```


```python
# Calculating the confusion matrix for the Random Forest
cm_rf = metrics.confusion_matrix(y_val,y_rf_pred)

# Display of confusion matrix
metrics.ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x293e4aa0210>




    
![png](images/output_58_1.png)
    



```python
target_labels = ['opinion', 'claim']
print(metrics.classification_report(y_val, y_rf_pred, target_names=target_labels))
```

                  precision    recall  f1-score   support
    
         opinion       1.00      1.00      1.00      1892
           claim       1.00      1.00      1.00      1925
    
        accuracy                           1.00      3817
       macro avg       1.00      1.00      1.00      3817
    weighted avg       1.00      1.00      1.00      3817
    
    

XGBoost:


```python
y_xgb_pred=clf_xgb.best_estimator_.predict(x_val)
```


```python
# Calculating the confusion matrix for the Random Forest
cm_xgb = metrics.confusion_matrix(y_val,y_xgb_pred)

# Display of confusion matrix
metrics.ConfusionMatrixDisplay(confusion_matrix=cm_xgb).plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x293e235e790>




    
![png](images/output_62_1.png)
    



```python
print(metrics.classification_report(y_val, y_xgb_pred, target_names=target_labels))
```

                  precision    recall  f1-score   support
    
         opinion       0.99      1.00      0.99      1892
           claim       1.00      0.99      0.99      1925
    
        accuracy                           0.99      3817
       macro avg       0.99      0.99      0.99      3817
    weighted avg       0.99      0.99      0.99      3817
    
    

Both models are able to obtain near perfect results, but the Random Forest manages to eek out a victory with a marginally higher recall evaluation score

### Testing

Using the best model to test never-seen-before data as a predictor for future behaviour


```python
y_pred = clf.best_estimator_.predict(x_test)
```


```python
# Calculating the confusion matrix for the Random Forest
cm_rf = metrics.confusion_matrix(y_test,y_pred)

# Display of confusion matrix
metrics.ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x293e2390a10>




    
![png](images/output_68_1.png)
    


#### Feature importances of the best model


```python
importances = clf.best_estimator_.feature_importances_

rf_importances = pd.Series(importances, index=x_test.columns).sort_values(ascending=False)[:10]

sns.barplot(x=rf_importances.index,y=rf_importances)
plt.xticks(rotation=45,ha='right')
plt.show()
```


    
![png](images/output_70_0.png)
    


The most predictive features are all related to the popularity of video. This is not unexpected, as analysis from prior EDA pointed to this conclusion.

## Conclusion

The models successfully demonstrate the potential of machine learning to automate the detection of video claims in TikTok, with the best-performing model, a Random Forest, providing near-perfect classification of claims.

## Recommendations:

* **Implement the predictive model**: The model can be integrated into the moderation pipeline to automatically flag claim-based content for faster reviews.

* **Combine predictive models with human moderators**: While the model performs well, the automated system should still be complemented by human reviewers, especially for more situational cases.

## Future Steps

* Expand the dataset to include multiple languages so it can work across the entirety of the content.

* Continue to improve the model with the feedback from moderators to include more subjective cases and improve performance over time.
