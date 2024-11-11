import pandas as pd
from fuzzywuzzy import fuzz

f = pd.read_csv('task1\\dataset.csv')
print("Task one : Movie Genere classification")
plot = input("Enter a plot description to search: ")
terms = plot.split()

match = None
hscore = 0

for i, row in f.iterrows():
    description = row['Description'].lower()
    movie_name = row['Movie_name']
    genere = row['genere']
    description = row['Description']
    total = sum(fuzz.partial_ratio(term, description) for term in terms)
    
    if total > hscore:
        hscore = total
        match = movie_name
        match2= genere
        match3= description

if match:
    print("predicted Movie : " , match)
    print("predicted Genere : " , match2)
    print("predicted Plot : ", match3)
else:
    print("No matching movies found.")
