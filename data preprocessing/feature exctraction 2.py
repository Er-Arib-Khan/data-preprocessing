from sklearn.feature_extraction import DictVectorizer

data = [
   {'age': 30, 'gender': 'male'},
   {'age': 25, 'gender': 'female'},
   {'age': 35, 'gender': 'male'}
]

vectorizer = DictVectorizer(sparse=False)
features_matrix = vectorizer.fit_transform(data)
feature_names = vectorizer.get_feature_names_out()

print("Feature Names:", feature_names)
print(features_matrix)
