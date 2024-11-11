import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fuzzywuzzy import fuzz
reader = pd.read_csv('task1\\sample_dataset.csv')
TFIDF = TfidfVectorizer(stop_words='english')
X = TFIDF.fit_transform(reader['Description']) 
target = reader['genere'] 
model = LogisticRegression(max_iter=1000)
model.fit(X, target)
def predictmovie(plot):
    vectorize = TFIDF.transform([plot]) 
    pgenre = model.predict(vectorize)[0]  
    genre_filtered_df = reader[reader['genere'] == pgenre].copy()
    genre_filtered_df.loc[:, 'score'] = genre_filtered_df['Description'].apply(lambda desc: fuzz.ratio(plot.lower(), desc.lower()))
    perfmatch = genre_filtered_df.loc[genre_filtered_df['score'].idxmax()]  
    matched_plot = perfmatch['Description']    
    return pgenre
plot = ("  ") #userinput
print("The plot You passed to the machine : " , plot )
pgenre  = predictmovie(plot)
print("Predicted Genre:", pgenre)
