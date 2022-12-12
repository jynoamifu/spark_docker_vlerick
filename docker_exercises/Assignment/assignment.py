# Import the necessary modules
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Create a SparkSession object
BUCKET = "dmacademy-course-assets"
KEY1 = "vlerick/pre_release.csv"
KEY2 = "vlerick/after_release.csv"

config = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Read the CSV file from the S3 bucket
variables = spark.read.csv(f"s3a://{BUCKET}/{KEY1}", header=True)
target = spark.read.csv(f"s3a://{BUCKET}/{KEY2}", header=True)

variables.show()
target.show()

# Create Pandas DataaFrame
variables = variables.toPandas()
target = target.toPandas()

print(variables.dtypes)
print(target.dtypes)

###### ML Model #######

# Inner Join the two tables
df_raw = pd.merge(variables, target, how='inner', on='movie_title')
print(df_raw.head())

# Delete unused target variables
df_raw = df_raw.drop(["num_critic_for_reviews" , "gross", "num_user_for_reviews", 
                      "imdb_score", "movie_facebook_likes"], axis = 1)
df_prep = df_raw

print(df_prep.dtypes)

# Check rows with duplicate movie titles and remove
df_prep = df_prep.drop_duplicates(subset = "movie_title",keep = "first")

# Delete columns that are not relevant for predicting our target variable. 
df_prep = df_prep.drop(["director_name", "actor_1_name", "actor_2_name","actor_3_name", "movie_title"], axis = 1)

# For the numerical values, alle facebook likes columns have a missing value and we will fill it in using the median
num_prep = df_prep.select_dtypes(include = {"float", "int"}).columns.tolist()
for i in num_prep: 
    df_prep[i] = df_prep[i].fillna(df_prep[i].median())
# Thus, we impute the missing value with the language "English"
df_prep.language.fillna(value = "English", inplace = True) 

# Fill in the most frequent value for the missing values in the categorical variable content rating 
df_prep["content_rating"] = df_prep["content_rating"].fillna(df_prep["content_rating"].mode().iloc[0])

print(df_prep.dtypes)

print(len(df_prep[df_prep.duration < 65]))
print(len(df_prep[df_prep.duration > 155]))

# Replace the outliers for the variable duration with the median value
for row in df_prep["duration"]:
    if df_prep["duration"] < 65 or df_prep["duration"] > 155:
        df_prep["duration"] = df_prep["duration"].median
    else:
        df_prep.duration = df_prep.duration 

print(len(df_prep[df_prep.duration < 65]))
print(len(df_prep[df_prep.duration > 155]))

# Create new column
df_prep["non_lead_actor_fb_likes"] = df_prep["actor_2_facebook_likes"] + df_prep["actor_3_facebook_likes"]

# Delete original columns
df_prep = df_prep.drop(["actor_2_facebook_likes", "actor_3_facebook_likes"], axis = 1)
# Drop column
df_prep = df_prep.drop("language", axis = 1)

# Create a new column with the number of genres 
df_prep["num_genres"] = list(map(lambda x: x.count("|") + 1, df_prep.genres))

# We create three groups: USA, UK and others by selecting the top 2 counts
count_country = round(df_prep.country.value_counts() / len(df_prep) * 100, 3)
vals_c = count_country[:2].index
df_prep['country'] = df_prep.country.where(df_prep.country.isin(vals_c), 'other_genres')

# Group content rating
count_rating = round(df_prep.content_rating.value_counts()/ len(df_prep) * 100, 3)
vals_cr = count_rating[:3].index
df_prep['content_rating'] = df_prep.content_rating.where(df_prep.content_rating.isin(vals_cr), 'other_content_rating')

# Create a new dataframe by separating the column genres by | 
df_gen = df_prep["genres"].str.split("|", expand = True)
# Look at all unique genres to find out which columns to create 
df_gen = df_gen.fillna("0")
# Create a list with the new column names
genre_columns = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
       'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
       'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance',
       'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
# Create new column for every genre in the original dataset and for every row give as value True or False
for value in genre_columns:
    df_prep[value] = df_prep.apply(lambda row: pd.notnull(row['genres']) and value in row['genres'].split('|'), axis=1)
# Drop original genre columns
df_prep = df_prep.drop("genres", axis = 1)

# Count percentage of genre occurence in a movie and sort
occurence = round(((df_prep[genre_columns] == True).sum()/ len(df_prep)) * 100, 2)
# Select top 9 genres: genres with an occurence of at least 10% so group other 13
vals_g = occurence[9:].index
# Create new grouped column
df_prep["other_genres"] = df_prep[vals_g].any(axis = "columns")
# Drop original genre columns with an occurance lower than 6% 
df_prep = df_prep.drop(vals_g.tolist(), axis = 1)

# Create duumy variables
df_prep = pd.concat([df_prep, pd.get_dummies(df_prep[["country", "content_rating"]])], axis = 1)
# Delete orginam columnsµ
df_prep = df_prep.drop(["country", "content_rating"], axis = 1)

# Create list of Boolean variables
boolean = df_prep.select_dtypes(include = 'bool').columns.tolist()
# Change Boolean variables to numeric variables where True = 1 and False = O
df_prep[boolean] = df_prep[boolean].astype("int")

df = df_prep
print(df)
print(df.dtypes)

df["num_voted_users"] = pd.cut(df["num_voted_users"], bins=["0", "20000", "60000","100000"], right = True, labels = False) + 1
print(df.shape)
print(df.duration)
print(df.num_voted_users)
# Define dependent and independent variable
x = df[df.loc[:, df.columns != "num_voted_users"].columns.tolist()]
# extract target variable
y = df["num_voted_users"]


# Define scaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

# randomly split into training (70%) and val (30%) sample
from sklearn.model_selection import train_test_split
seed = 123 
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.3, random_state = seed)

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

# Random Forest
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini
rfc.fit(x_train, np.ravel(y_train,order='C'))
rfcpred = rfc.predict(x_val)
cnf_matrix = metrics.confusion_matrix(y_val, rfcpred)
print(cnf_matrix)
print("Accuracy:",metrics.accuracy_score(y_val, rfcpred))