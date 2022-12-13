# Import the necessary modules
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
import numpy as np
import os


# Create a SparkSession object
BUCKET = "dmacademy-course-assets"
KEY1 = "vlerick/pre_release.csv"
KEY2 = "vlerick/after_release.csv"

if 'AWS_SECRET_ACCESS_KEY' in os.environ:
    print("present in environment")
    config = {
        "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
        "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    }
else:
    print("not environment")
    config = {
        "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
        "spark.hadoop.fs.s3a.aws.credentials.provider": "com.amazonaws.auth.InstanceProfileCredentialsProvider",
    }

conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Read the CSV file from the S3 bucket
pre_release = spark.read.csv(f"s3a://{BUCKET}/{KEY1}", header=True)
after_release = spark.read.csv(f"s3a://{BUCKET}/{KEY2}", header=True)

pre_release.show()
after_release.show()

#Convert the Spark DataFrames to Pandas DataFrames.

pre_df = pre_release.toPandas()
after_df = after_release.toPandas()

#Merging the two data frames based on the movie_title column as this will be necessary for our prediction
#Checking the first few rows to see the effect of the merge
df = pd.merge(pre_df, after_df, how='inner', on='movie_title')
df.head(5)
# we drop the variables we don't need
df = df.drop(columns = ["gross","num_critic_for_reviews","num_voted_users",
                       "num_user_for_reviews","movie_facebook_likes","director_name","actor_1_name","actor_2_name",
                        "actor_3_name","movie_title","actor_1_facebook_likes","actor_2_facebook_likes",
                        "actor_3_facebook_likes"])
# drop content rating
df = df.drop("content_rating", axis='columns')
# we delete all rows with missing values
df = df.dropna()
df.shape
# we delete the duplicates and immediately update the dataframe
df.drop_duplicates(inplace = True)
df.shape
# we see that the 22 duplicates have been removed since the number of observations decreased to 1044.
# we now replace all languages that are not English, French or Spanish with 'other language'
vals = df["language"].value_counts()[:3].index
print (vals)
df['language'] = df.language.where(df.language.isin(vals), 'other_language')
# we use the OneHotEncoder function to encode the language variable into dummies
# we turn language into dummies and check if it worked
ohe = OneHotEncoder()
df_1 = ohe.fit_transform(df[['language']])
df[ohe.categories_[0]] = df_1.toarray()
df.tail(10)
# we repeat this process for country
vals = df["country"].value_counts()[:6].index
print (vals)
df['country'] = df.country.where(df.country.isin(vals), 'other_country')
# we use OneHotEncoder again
ohe = OneHotEncoder()
df_2 = ohe.fit_transform(df[['country']])
df[ohe.categories_[0]] = df_2.toarray()
df.tail(10)
# we extract dummies from the genres column, by separating different string first, then we combine the newly
# created dummies and concatenate them with original dataset
df_dumm = df['genres'].str.get_dummies(sep = '|')
comb = [df, df_dumm]
df = pd.concat(comb, axis = 1)
# sum all genres except for the most common ones
df["other_genres"] = df["Animation"]+df["Biography"]+df["Documentary"]+df["Film-Noir"]
+df["History"]+df["Music"]+df["Musical"]+df["Mystery"]+df["Sci-Fi"]+df["Short"]
+df["Sport"]+df["War"]+df["Western"]
# now we replace their values by 1
df=df.replace(2,1)
df=df.replace(3,1)
# now we can delete the non-common genres
df = df.drop(columns = ["Animation","Biography","Documentary","Film-Noir","History","Music",
             "Musical","Mystery","Sci-Fi","Short","Sport","War","Western"])
df = df.drop(columns=["genres","language","country"])
# But before we can start building models, we first have to extract the target variable and 
# the explanatory variables
# for multicollinearity reasons, we also drop the country variables, as they might be correlated with language.
x = df.drop(columns = ["imdb_score","USA","UK","France","Canada","Germany","Australia","other_country"])
y = df["imdb_score"]
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size = 0.25,
                                                 random_state=42)

print(X_train.shape)
print(X_test.shape)
def accuracy_cont(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))
    accuracy = 1-mape
    return accuracy
# train model
rf = RandomForestRegressor(max_depth=10, min_samples_leaf =1, random_state=0)
rf.fit(X_train, y_train)
#predict regression forest 
array_pred = np.round(rf.predict(X_test),0)
#add prediction to data frame
y_pred = pd.DataFrame({"y_pred": array_pred},index=X_test.index)
val_pred = pd.concat([y_test,y_pred,X_test],axis=1)
val_pred
#Evaluate model
#by comparing actual and predicted value 
act_value = val_pred["imdb_score"]
pred_value = val_pred["y_pred"]

print(pred_value.dtype)
print(val_pred)

# 4: convert back to pyspark dataframe
pred_final = spark.createDataFrame(val_pred)

# 5 write to S3 bucket as JSON lines 
pred_final.write.json(f"s3a://{BUCKET}/vlerick/noamifu/", mode="overwrite")
