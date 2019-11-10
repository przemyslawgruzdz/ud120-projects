from nltk.corpus import stopwords
sw = stopwords.words('english')
print ('Number of stopwords: {0}'.format(len(sw)))

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
print stemmer.stem('responsiveness')
print stemmer.stem('responsivity')
print stemmer.stem('unresponsive')