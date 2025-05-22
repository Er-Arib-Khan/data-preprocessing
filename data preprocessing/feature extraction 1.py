from sklearn.feature_extraction.text import CountVectorizer
documents =[
    "your service is very very bad",
    "tcs is service based company",
    "you work in bad service company"
]
Count_Vectorizer = CountVectorizer()
count_matrix = Count_Vectorizer.fit_transform(documents)
print("vocabulary:",Count_Vectorizer.vocabulary_)
print(count_matrix.toarray())
