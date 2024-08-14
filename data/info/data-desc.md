# ELLIPSE Corpus Dataset Description

## Overview

The ELLIPSE corpus is a dataset of argumentative essays written by 8th-12th grade English Language Learners (ELLs). These essays have been scored on six analytic measures, each representing a component of proficiency in essay writing.

## Scoring Measures

Essays are scored on the following six measures:

1. Cohesion
2. Syntax
3. Vocabulary
4. Phraseology
5. Grammar
6. Conventions

Scores for each measure range from 1.0 to 5.0 in increments of 0.5, with higher scores indicating greater proficiency.

## Task

The primary task is to predict the score for each of the six measures for essays provided in the test set.

## Dataset Origin

Some essays in this dataset have appeared in previous competitions:
- Feedback Prize - Evaluating Student Writing
- Feedback Prize - Predicting Effective Arguments

Participants are encouraged to utilize these earlier datasets in this competition.

## File and Field Information

### train.csv
- Contains the training set
- Fields:
  - `text_id`: Unique identifier for each essay
  - `full_text`: Complete text of the essay
  - Scores for each analytic measure: `cohesion`, `syntax`, `vocabulary`, `phraseology`, `grammar`, `conventions`

### test.csv
- Contains the test set
- Fields:
  - `text_id`: Unique identifier for each essay
  - `full_text`: Complete text of the essay
- Note: The provided file contains only a few sample essays

### sample_submission.csv
- A submission file in the correct format
- Refer to the Evaluation page for detailed information

## Important Notes

1. The `test.csv` file provided initially contains only sample essays to help with solution development.
2. During final scoring, the sample test data will be replaced with the full test set. The full test set contains approximately 2,700 essays.