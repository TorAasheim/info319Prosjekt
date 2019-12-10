import codecs
import json
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
nltk.download('stopwords')
import re

# Reads the json file with the data
with open("tweets.json", encoding='utf-8') as file:
    data = json.load(file)

# saves all the tweets and dates in a list
tweet = []
for x in data:
    tweet.append(str(x['tweets']) +'; '+ str(x['time']))

# stemming and removing comma
ps = PorterStemmer()
stemTweet = []
for x in tweet:
    stemTweet.append(ps.stem(x))
    x.replace(',', ' ')


cleanTweet = []
#Regex for date
reg = '(0?[1-9]|1[0-2])[\/](0?[1-9]|[12]\d|3[01])[\/](19|20)\d{2}'
#Remove timestamp, so that we are only left with the data
for x in stemTweet:
    condition = re.search(reg, x)
    if condition:
        str = re.search(reg, x)
        start = str.span()[1]
        stopp = x[-1]
        cleanTweet.append(x[0:start])


#Adds a header to the list
cleanTweet.insert(0,'text' +';'+ 'time' )
#Writes to file
file_out = codecs.open("cloud.csv", "w", "utf-8")
for x in cleanTweet:
    file_out.write(x + '\n')
file_out.close()
