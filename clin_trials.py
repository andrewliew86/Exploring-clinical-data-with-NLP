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