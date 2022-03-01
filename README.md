
# Russia-Ukraine War Sentiment Analysis on Tweets
On 23rd February, Russia announced special military operation on Ukraine. This geopolitical incident has been put under the limelight. As people gather on social media to express their opinion on the incident, it will be insightful to analyse their sentiment in an attempt to understand the public opinion.


```python
import pandas as pd
import matplotlib.pyplot as plt
import re
import yaml
import tweepy
import numpy as np
from wordcloud import WordCloud
from textblob import TextBlob
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

## Collecting tweets
We use Tweepy which is built on Twitter API to collect tweets. Basic data cleaning is done on data, such as removing URL to minimise its effect on extracting the sentiment of user. 


```python
with open('config.yaml') as file:
    keys = yaml.safe_load(file)
    consumer_key = keys['search_tweets_api']['consumer_key']
    consumer_secret = keys['search_tweets_api']['consumer_secret']
    access_token = keys['search_tweets_api']['access_token']
    access_token_secret = keys['search_tweets_api']['access_token_secret']
```


```python
def auth():
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
    except:
        print('An error occurred during the authentication')
    return api
```


```python
# function to remove URL
def remove_url(txt):
    return ' '.join(re.sub('([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', txt).split())

# function to remove time
def remove_time(datetime):
    return str(datetime).split(' ')[0]
```


```python
def search_by_hashtag(api, date_until, words, max):
    df = pd.DataFrame(columns=['id', 'created_at', 'username', 'location', 'following', 
                               'followers', 'retweetcount', 'text']) 
    tweets = tweepy.Cursor(api.search_tweets, q=words, lang='en', until=date_until, tweet_mode='extended').items(max) 
    list_tweets = [tweet for tweet in tweets] 
    
    for tweet in list_tweets: 
        id = tweet.id
        created_at = remove_time(tweet.created_at)
        username = tweet.user.screen_name 
        location = tweet.user.location 
        following = tweet.user.friends_count 
        followers = tweet.user.followers_count  
        retweetcount = tweet.retweet_count 
    
        try: 
            text = tweet.retweeted_status.full_text 
        except AttributeError: 
            text = tweet.full_text 
        text = remove_url(text)

        tweets = [id, created_at, username, location, following, followers, retweetcount, text]

        df.loc[len(df)] = tweets # add current tweet to the last
          
    return df
```


```python
api = auth()
words = 'Ukraine Russia -filter:retweets'
date_until = '2022-02-25'
df = search_by_hashtag(api, date_until, words, 100000)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>created_at</th>
      <th>username</th>
      <th>location</th>
      <th>following</th>
      <th>followers</th>
      <th>retweetcount</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1496635945155960835</td>
      <td>2022-02-23 23:59:50+00:00</td>
      <td>DafinStudio</td>
      <td>NaN</td>
      <td>200</td>
      <td>67</td>
      <td>1</td>
      <td>I Stand With Ukraine TShirtUkraineRussiaCrisis...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1496635862574321669</td>
      <td>2022-02-23 23:59:30+00:00</td>
      <td>martinigrimaldi</td>
      <td>NaN</td>
      <td>97</td>
      <td>10525</td>
      <td>0</td>
      <td>Titles that will make history Japan imposes sa...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1496635580532211712</td>
      <td>2022-02-23 23:58:23+00:00</td>
      <td>anainkc</td>
      <td>Kansas City, MO</td>
      <td>66</td>
      <td>490</td>
      <td>7</td>
      <td>AND JUST LIKE THAT INFLATION THE supplychaincr...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1496635414517481472</td>
      <td>2022-02-23 23:57:43+00:00</td>
      <td>keddle01</td>
      <td>NaN</td>
      <td>5026</td>
      <td>5112</td>
      <td>1020</td>
      <td>The Simpsons predicted theCrisis of Putin Russ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1496635050888208388</td>
      <td>2022-02-23 23:56:16+00:00</td>
      <td>Macid3000</td>
      <td>51.46386,0.006213</td>
      <td>4825</td>
      <td>2211</td>
      <td>8886</td>
      <td>Putin convened an unscheduled meeting with his...</td>
    </tr>
  </tbody>
</table>
</div>



## Sentiment Analysis
### Microsoft Azure
We adopt Text Analytics in Azure Cognitive Services to obtain the sentiment score of tweets. The API can ouput confidence scores of the text being postive, neutral and negative respectively. For consistency and easier understanding, we will map the scores into a composite score in range [-1. 1].


```python
# Authenticate the client using your key and endpoint 
def authenticate_client():
    with open('config.yaml') as file:
        keys = yaml.safe_load(file)
        key = keys['azure']['subscription_key']
        endpoint = keys['azure']['endpoint']
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    return text_analytics_client
```


```python
# Function for detecting sentiment in text
def sentiment_analysis_example(client):

    documents = list(df['text'])
    polarity = []
    azure_class = []
    # For loop is used to bypass the batch request limit
    for document in documents:
        response = client.analyze_sentiment(documents=[document])[0]
        azure_class.append(response.sentiment)
        polarity.append(0 + response.confidence_scores.positive - response.confidence_scores.negative)
    df['azure_polar'] = polarity
    df['azure_class'] = azure_class

client = authenticate_client()      
sentiment_analysis_example(client)
```

### TextBlob
Textblob is an open-source python library for processing textual data. It can evaluate both polarity and subjectivity in text. The polarity score is a float within the range [-1.0, 1.0]. The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.


```python
sentiment_objects = [TextBlob(tweet) for tweet in list(df['text'])]
blob_polar = [tweet.sentiment.polarity for tweet in sentiment_objects]
blob_subj = [tweet.sentiment.subjectivity for tweet in sentiment_objects]
df['blob_polar'] = blob_polar
df['blob_subj'] = blob_subj
```

### Results


```python
df.to_csv('processed_tweets.csv')
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>created_at</th>
      <th>username</th>
      <th>location</th>
      <th>following</th>
      <th>followers</th>
      <th>retweetcount</th>
      <th>text</th>
      <th>azure_polar</th>
      <th>azure_class</th>
      <th>blob_polar</th>
      <th>blob_subj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1496635945155960835</td>
      <td>2022-02-23 23:59:50+00:00</td>
      <td>DafinStudio</td>
      <td>NaN</td>
      <td>200</td>
      <td>67</td>
      <td>1</td>
      <td>I Stand With Ukraine TShirtUkraineRussiaCrisis...</td>
      <td>-0.96</td>
      <td>negative</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1496635862574321669</td>
      <td>2022-02-23 23:59:30+00:00</td>
      <td>martinigrimaldi</td>
      <td>NaN</td>
      <td>97</td>
      <td>10525</td>
      <td>0</td>
      <td>Titles that will make history Japan imposes sa...</td>
      <td>-0.05</td>
      <td>neutral</td>
      <td>0.0</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1496635580532211712</td>
      <td>2022-02-23 23:58:23+00:00</td>
      <td>anainkc</td>
      <td>Kansas City, MO</td>
      <td>66</td>
      <td>490</td>
      <td>7</td>
      <td>AND JUST LIKE THAT INFLATION THE supplychaincr...</td>
      <td>0.01</td>
      <td>neutral</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Further Analysis


```python
df = pd.read_csv('processed_tweets.csv')
```

We first compare the sentiment scores rated by TextBlob and Azure respectively. It can be seen from the figure that Azure has a wider spectrum of sentiment, suggesting that it uses a more advanced algorithm to analyse the sentiment in text.


```python
plt.figure(figsize=(8, 6), dpi=80)

ax1 = plt.subplot(2, 1, 1)
df.hist(column='blob_polar', bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1], ax=ax1)
ax1.set_title('Sentiments analysed by TextBlob')

ax2 = plt.subplot(2, 1, 2)
df.hist(column='azure_polar', bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1], ax=ax2)
ax2.set_title('Sentiments analysed by Azure')
```




    Text(0.5, 1.0, 'Sentiments analysed by Azure')




![png](sentiment_files/sentiment_19_1.png)



```python

df['azure_class'].value_counts().plot(kind='bar', title='Sentiment Class of Tweets on Ukraine-Russia War')
```




    <AxesSubplot:title={'center':'Sentiment Class of Tweets on Ukraine-Russia War'}>




![png](sentiment_files/sentiment_20_1.png)


Regarding the sentiment of tweets, we can see a general trend towards negative sentiment, which is in line with the common belief that people do not like war. To understand the general opinion better, we can look at key words presented in each sentiment class.


```python
fig = plt.figure(figsize=(24, 72), dpi=300)
for index, sentiment in enumerate(['positive', 'neutral', 'negative']):
    words = []
    tokens = [str(sentence).split() for sentence in list(df[df['azure_class'] == sentiment]['text'])]
    for i in range(len(tokens)):
        for word in tokens[i]:
            word = word.lower()
            words.append(word)
    wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate(' '.join(words))
    ax = plt.subplot(1, 3, index+1)
    ax.imshow(wordcloud)
    ax.set_title(f'Words in tweets classified as {sentiment}')
    ax.axis("off")
```


![png](sentiment_files/sentiment_22_0.png)


Some remarkable keywords in respective classes include:
- Positive: peace, support, independent
- Neutral: mostly proper nouns
- Negative: crisis, war, invasion

Aside from the aggregate analysis, we can also look at the change in sentiment of users.


```python
positive_count = []
neutral_count = []
negative_count = []
days = np.unique(list(map(lambda x: x.split(' ')[0], df['created_at'])))
for day in days:
    current = df[df['created_at'].str.contains(day)]
    positive_count.append(current[current['azure_class'] == 'positive']['id'].count())
    neutral_count.append(current[current['azure_class'] == 'neutral']['id'].count())
    negative_count.append(current[current['azure_class'] == 'negative']['id'].count())
count = pd.DataFrame({'Positive': positive_count, 'Neutral': neutral_count, 'Negative': negative_count}, index=days)
count.plot(title='Change in Tweet Sentiment')
plt.show()
```


![png](sentiment_files/sentiment_25_0.png)


At last, we examine the subjectivity of tweets.


```python
fig, ax = plt.subplots()
df.hist(column='blob_subj', bins=[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1], ax=ax)
ax.set_title('Subjectivity from Tweets on Ukraine-Russia War')
```




    Text(0.5, 1.0, 'Subjectivity from Tweets on Ukraine-Russia War')




![png](sentiment_files/sentiment_27_1.png)


It can be seen that most users are objective, which is unusual on social media. This can be attributed to the fact that people tend to express objective opinion while sharing news.
