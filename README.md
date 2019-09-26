Supporting code & data for "Summarizing Relationships for Interactive Concept Map Browsers"
"New Frontiers in Summarization" workshop at EMNLP 2019

#### Setup environment
$ source activate maps && pip install -r requirements.txt

#### Build dataset

- Clone repo
- Run `.publish/make_dataset.sh`

#### Make results 
- Running models, `python modeling/run_model.py`
- Running IAA, `python code/run_iaa.py`
- Get numbers in paper, `python numbers_in_paper.py`
