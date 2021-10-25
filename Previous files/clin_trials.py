# We will be analyzing clinical trial text data using topic modelling and named entity recognition 

# First get the clinical trial data into a dataframe
# We use pytrials, a Python wrapper around the clinicaltrials.gov API.
from pytrials.client import ClinicalTrials
import pandas as pd

# You create an instance of the ClinicalTrials() object 
ct = ClinicalTrials()

# Here is a list of the fields that would be returned in your results
fields = ["NCTId",
          "PrimaryCompletionDate",
          "BriefSummary", 
          "BriefTitle",
          "ResponsiblePartyInvestigatorAffiliation",
          "ResponsiblePartyInvestigatorFullName",
          "PrimaryOutcomeDescription",
          "Phase",
          "Condition",
          "ConditionMeshTerm",
          "LocationCountry"
            ]

# Note, the maximum rows that can be returned is only 1000!
data = ct.get_study_fields("Coronavirus+COVID", fields, max_studies=1000, fmt='csv')

# Convert to a dataframe and save as a csv file
clinical_trials = pd.DataFrame.from_records(data[1:], columns=data[0])
clinical_trials.to_csv("clinical_trials_nlp.csv")

#%%
# See https://towardsdatascience.com/building-a-topic-modeling-pipeline-with-spacy-and-gensim-c5dc03ffc619
# See also: https://medium.com/@colemiller94/topic-modeling-with-spacy-and-gensim-7ecfd3de95f4
# For preprocessing of abstracts: https://medium.com/@omar.abdelbadie1/processing-text-for-topic-modeling-c355e907ab23
# Let's perform some text preprocessing and then 
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language
import re
from tqdm import tqdm
import string  # For punctuation removal


# Load the Spacy model
nlp= spacy.load("en_core_web_sm")

# Lets take of stopwords first
# Add custom list of stopwords to the default list.
stop_list = ['trial', "covid", "covid-19", "sars-cov-2", "s"]

# Updates spaCy's default stop words list with my additional words. 
nlp.Defaults.stop_words.update(stop_list)

# Adding custom stopwords means that we have to reset the "is_stop" flag.
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True


# Now, create pipeline component for stop word removal
# Note that lemmatization is included in the default Spacy v 3.0 pipeline
# Setup regex pattern for removing punctuation (except hyphens)
remove = string.punctuation
remove = remove.replace("-", "") # don't remove hyphens
pattern = r"[{}]".format(remove) # create the pattern    

@Language.component("remove_stopwords")    
def remove_stopwords(doc):
    """ Removes punctuation and stopwords from a text input and returns strings (for Gensim analysis). Note that we are excluding all punctuation except dashes which has been shown to be important for scientific documents"""
    # Use token.text to return strings, which we'll need for Gensim.
    doc = [token.text for token in doc if token.is_stop != True]
    # Use pattern above to remove punctuation except hyphens
    doc = [re.sub(pattern, " ", token) for token in doc]
    return doc


# The add_pipe function appends our function to the default pipeline.
nlp.add_pipe("remove_stopwords", last=True)


#nlp.add_pipe("info_component", name="print_info", last=True)


#%%
# Create a document list
df = pd.read_csv('clinical_trials_nlp.csv')
my_docs = df['BriefSummary']

doc_list = []
# Iterates through each article in the corpus.
for doc in tqdm(my_docs):
    # Passes that article through the pipeline and adds to a new list.
    pr = nlp(doc)
    doc_list.append(pr)

#%%
import re
import numpy as np
import pandas as pd

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_multiple_whitespaces
from gensim.models import Phrases
from gensim.models.phrases import Phraser


# spacy for lemmatization
import spacy
# nltk for stopwords
from nltk.corpus import stopwords


# Lets take of stopwords first
# Add custom list of stopwords to the default list.
stop_words = stopwords.words('english')
stop_words.extend(['trial', "covid", "covid-19", "sars-cov-2", "s"])

# Convert my dataframe column containing the text for modelling to a list
data = df['BriefSummary'].values.tolist()

# Optional: remove new line characters
#data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove punctuation manually (to ensure that we keep the hyphen which is important for scientific documents)
remove = string.punctuation
remove = remove.replace("-", "")  # don't remove hyphens
pattern = r"[{}]".format(remove)  # create the pattern    
data = [re.sub(pattern, " ", sent) for sent in data]  # Use pattern to remove punctuation except hyphens

# Tokenize sentences into words, and use gensim to strip whitespace and remove stop_words
custom_gensim_filters = [lambda x: x.lower(), strip_multiple_whitespaces(x), remove_stopwords(x, stopwords=stop_words)]
data_words = [preprocess_string(sent, custom_gensim_filters) for sent in data]


# What if we want to get bigrams and trigrams?
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=3, threshold=50) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=50)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

print(trigram_mod[bigram_mod[data_words[0]]])


#%%

# see: https://towardsdatascience.com/building-a-topic-modeling-pipeline-with-spacy-and-gensim-c5dc03ffc619
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# Gensim libraries
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import re
import pandas as pd
import string  # For punctuation removal

# Add custom stop words
sw_nltk = stopwords.words('english')
sw_nltk.extend(['trial', 
                "covid", 
                "covid-19", 
                "sars-cov-2",
                "coronavirus",
                "study",
                "infection",
                "s"])

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Preprocess text in dataframe column (stopword removal, lowercase, punctuation removal[Note: no removal of / or -])"""
    # First split the text based on whitespace
    words = [word for word in text.split()]
     # Setup custom removal of punctuation (i.e. remove all but keep hyphens and backlash as these are important in scientific studies)
    remove = string.punctuation
    remove = remove.replace("-", "")  # don't remove hyphens
    remove = remove.replace("/", "")  # don't remove backslash
    pattern = r"[{}]".format(remove)  # create the pattern    
    words = [re.sub(pattern, "", w) for w in words]
    #lemmatize words and remove stopwords
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in sw_nltk]
    return words


# Read clinical trial data
df = pd.read_csv('clinical_trials_nlp.csv')

    
# Apply the preprocess function to the 'BriefSummary' column and return as a new column
df["Summary_tokens"] = df['BriefTitle'].apply(lambda x: preprocess(x))
# Create a list of documents from the column for preprocessing
doc_list = df['Summary_tokens'].to_list()

# Map word IDs to words.
words = gensim.corpora.Dictionary(doc_list)

# Turns each document into a bag of words.
corpus = [words.doc2bow(doc) for doc in doc_list]

#%%
# LDA model tips here: https://towardsdatascience.com/6-tips-to-optimize-an-nlp-topic-model-for-interpretability-20742f3047e2
# also here: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=words,
                                           num_topics=10, 
                                           random_state=2,
                                           update_every=1,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print out the topics and their scores from the LDA model
for i, v in lda_model.print_topics(num_words=10):
    print("topic {}:".format(i))
    print(v)
    
    
#%%
# Apart from the bag of words model, lets also perform tf-idf vectorization and use the results for LDA modelling for comparrison
tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf,
                                           id2word=words,
                                           num_topics=10, 
                                           random_state=2,
                                           update_every=1,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print out the topics and their scores from the LDA model
for i, v in lda_model.print_topics(num_words=10):
    print("topic {}:".format(i))
    print(v)
    
#%%
# pyLDavis to visualize the topics
# pyLDavis is a wrapper aroud
import pyLDAvis
import pyLDAvis.gensim
#from IPython.core.display import HTML

vis = pyLDAvis.gensim.prepare(topic_model=lda_model, 
                              corpus=corpus_tfidf, 
                              dictionary=words)

#viz = pyLDAvis.display(vis)
pyLDAvis.show(vis)


#pyLDAvis.enable_notebook()
#HTML(pyLDAvis.display(viz))

#%%
# Can also plot coherence score to check if the appropriate number of topics is selected
coherence = []
for k in range(1,30):
    print('Round: '+str(k))
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(corpus=corpus_tfidf,
                    id2word=words,
                    num_topics=k, 
                    random_state=2,
                    update_every=1,
                    passes=10,
                    alpha='auto',
                    per_word_topics=True)
    
    cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, 
                                                     texts=corpus_tfidf,
                                                     dictionary=words, 
                                                     coherence='c_v')   
                                                
    coherence.append((k,cm.get_coherence()))



