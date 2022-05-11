# %% Import Library

from nltk import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk   # Natural Language Tool Kit
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
Eğer nltk kodalrı çalışmıyorsa lookup hatası alıyorsan;

nltk.download('popular')

komutunu kernelda çalıştır ve öyle dene.
"""
# %% Import Data

data = pd.read_csv("gender_classifier.csv", encoding="latin1")
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis=0, inplace=True)
data.gender = [1 if each == "female" else 0 for each in data.gender]

# %% Cleaning Data
"""
Çalışacağımız data, twitterda yazılan yorumlar ve  cinsiyetleri içeririyor.
bizz yazılan yorumdan cinsiyeti bulmaya çalışıyoruz.
"""

# Regular expression RE Öğreneceğiz. Mesela [ ^ a-z, A-Z] bu bir re

first_description = data.description[4]

# Ilk cleaning işlemi
description = re.sub("[^a-zA-Z]", " ", first_description)

"""
decription = re.sub([ ^ a-zA-Z], " ", first_description)

burada ne yazıyor:

a dan z ye ve A dan Z ye  kadar olanları alma. (^) bu işaret alma demek. sonra
virgül tırnak içinde bu bulduklarını bununla değiştir demek. yani " " yaparsak
boşlukla değiştir demek. Sonra virgül hangi description üzerinde uygulanacaksa
onun adı.
"""

# Ikinci cleaning islemi

"""
Şimdi bilgisayar dilinde THE ve the farklı kelimelerdir. Bu sebeple tüm datayı
lower yani küçük harfle yazacağız.
"""

description = description.lower()  # Büyük harften küçük harfe çevirme.


# %% Irrelavent words (Alakasız/gereksiz Kelimeler) Stopwords

"""
I go to to the school and home
şimdi burada "the" ve "and" kelimeleri irrelavent kelime çünkü bunlar gramerle
ilgili bizim classifier etmemize yardımcı değil. Bu cümleyi bir kadında bir
erkekte yazmış olabilir. Bunları datadan çıkarmamız gerek bunu yaparkende
nltk kullanacağız.
"""

nltk.download("stopwords")

# description = description.split()
# Split yerine tokenizer kullanabiliriz.

description = nltk.word_tokenize(description)
"""
Çalışacağımız data, twitterda yazılan yorumlar ve  cinsiyetleri içeririyor.
bizz yazılan yorumdan cinsiyeti bulmaya çalışıyoruz.
"""

# Regular expression RE Öğreneceğiz. Mesela [ ^ a-z, A-Z] bu bir re

first_description = data.description[4]

# Ilk cleaning işlemi
description = re.sub("[^a-zA-Z]", " ", first_description)

"""
decription = re.sub([ ^ a-zA-Z], " ", first_description)

burada ne yazıyor:

a dan z ye ve A dan Z ye  kadar olanları alma. (^) bu işaret alma demek. sonra
virgül tırnak içinde bu bulduklarını bununla değiştir demek. yani " " yaparsak
boşlukla değiştir demek. Sonra virgül hangi description üzerinde uygulanacaksa
onun adı.
"""

# Ikinci cleaning islemi

"""
Şimdi bilgisayar dilinde THE ve the farklı kelimelerdir. Bu sebeple tüm datayı
lower yani küçük harfle yazacağız.
"""

description = description.lower()  # Büyük harften küçük harfe çevirme.


# %% Irrelavent words (Alakasız/gereksiz Kelimeler) Stopwords

"""
I go to to the school and home
şimdi burada "the" ve "and" kelimeleri irrelavent kelime çünkü bunlar gramerle
ilgili bizim classifier etmemize yardımcı değil. Bu cümleyi bir kadında bir
erkekte yazmış olabilir. Bunları datadan çıkarmamız gerek bunu yaparkende
nltk kullanacağız.
"""

nltk.download("stopwords")

# description = description.split()
# Split yerine tokenizer kullanabiliriz.
description = nltk.word_tokenize(description)
"""
Peki ; description = nltk.word_tokenize(description) ; kullanmanın avantajı ne
neden split yerine buunu kullanmak iyidir. Çünkü ;
- Mesela diyelim ki bizim bir str miz var. Bu str "shouldn't ve guzel" ben bunu
split dediğim zaman : "shouldn't", "ve", "guzel" diyerek ayırır.
Ama "shouldn't" kelimesi aslında "should not" demek benim bunu
"should" ve "not" diye 2 ye ayırabilmem lazım. word_tokenize bunu yapabiliyor.

yani bu stringi "shouldn't ve guzel" = buna çeviriyor;

"should", "n't", "ve", "guzel"

bunun nedeni split sadece boşluklara göre ayırırı.
"""

# %% Gereksiz kelimleri çıkar.

# Bu kez list comprension kullanalım

description = [
    word for word in description if not word in set(
        stopwords.words("english"))]

# %% Kökleri  bulma (lemmatization)
# (lemmatization) = loved == love / gitmeyeceğim == git
"""
Bunu neden yapıyoruz. Örneğin :
    maça gitmek çok güzeldir
    maç çok iyidir
    maçı kazandık
buradaki göründüğü üzere bu karakterin cinsiyetini ayırt edebileceğimiz kelime
nedir?
Maç

maça maçı maç  bilgisayar için 3 farklı kelime biz o yüzden bunu böyle
tutmaktansa maç olarak öğretmek bizim algoritmamızın ve bilgisayarımızın işini
kolaylaştırı.
"""

lemma = nltk.WordNetLemmatizer()
description = [
    lemma.lemmatize(word) for word in description]  # Artık kökleri bullduk.

# Şimdi cümlemizi görmek adına bu köklerden oluşan listeyi joinleyelim

description = " ".join(description)

"""
Şu zamana kadarkileirn hepsi tek bir yorum datası içindi benim bunu tüm datama
uygulamam gerekiyor bunun içinde for döngüsü kullanacağız.
"""


# %% DATA CLEANİNG FOR ALL DATA

description_list = []

for description in data.description:

    description = re.sub("[^a-zA-Z]", " ", description)

    description = description.lower()

    description = nltk.word_tokenize(description)

    # description = [
    #     word for word in description if not word in set(
    #         stopwords.words("english"))]

    description = [
        lemma.lemmatize(word) for word in description]

    description = " ".join(description)

    description_list.append(description)
"""
burada
    # description = [
    #     word for word in description if not word in set(
    #         stopwords.words("english"))]
yapmamızın nedeni stopwordün çok uzun sürmesi.
"""


# %% Bag Of Word

# from sklearn.feature_extraction.text import CountVectorizer burada yapıldı.
"""
- 1 lerden ve 0 lardan oluşan bu bag of words tipine sparce_matrix denir.
- count_vectorizer = CountVectorizer(max_features=max_features,
                                   stop_words="english")
burada "english" den sonra "," koyup token_pattern yazarak tokenize yapmak
mümkün ama biz forda yaptığımız için buna gerek yok.

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

toarry yazmak görmek istememeizden kaynaklı.
"""

max_features = 5000
count_vectorizer = CountVectorizer(max_features=max_features,
                                   stop_words="english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
# Sparce matrix aslında bizim x imiz, yani featureların olduğu kısım.

# En sık kullanılan 500 kelimeye bakalım.

print("en sık kullanılan {} kelimeler : {}"
      .format(max_features, count_vectorizer.get_feature_names()))

# %% Text Classification

# Bize y lazım
x = sparce_matrix
y = data.iloc[:, 0].values  # Male or Female classes...

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.1,
                                                    random_state=42)
# Naive Bayes Kullanacağız,

nb = GaussianNB()

nb.fit(x_train, y_train)

# Prediction

y_pred = nb.predict(x_test)

print("accuracy = ", nb.score(y_pred.reshape(-1, 1), y_test))
