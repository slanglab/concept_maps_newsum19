'''
This loads display sets from MS Excel
'''

import xlrd
import json
from collections import defaultdict

class DisplaySet(object):
    '''
    A display set is a collection of sentences observed by ONE annotator associated w/ one mention set

    On the excel sheets, each individual DisplaySet is set apart by ***
    
    A DisplaySet is the same thing as what Saif Mohammad calls a "tuple"

    http://saifmohammad.com/WebPages/bwsVrs.html#SHR
    '''
    def __init__(self, annotator):
        self.annotator = annotator
        self.best = None
        self.worst = None
        self.mention_set = None
        self.sentences = []

    def validate(self):
        assert type(self.sentences) == list
        if len(self.sentences) > 1:
            # In a few rare cases there are tuples w/ only one sentence in them
            # In these cases the validation does not need to pass
            assert type(self.best) == type(self.worst) == unicode
            assert self.best in self.sentences
            assert self.worst in self.sentences
        assert type(self.mention_set) == unicode
        assert type(self.annotator) == int
        assert self.annotator in [1,2,3]


def get_display_sets_from_excel_sheet(file_name, annotator):
    '''
    inputs:
        file_name (string, pointer to excel sheet)
        annotator (int, which annotator)
    returns:
        {sentence:pick} for sentence in sentences
        pick is {"", "best", "worst"}

    '''
    assert annotator in [1,2,3]
    book = xlrd.open_workbook(file_name) 

    print("reading: {0}".format(book.sheet_names()))

    out = []

    with open("output/sent2mentionset.json", "r") as inf:
        sent2mentionset = json.load(inf)

    for sheet in range(book.nsheets):
        sh = book.sheet_by_index(sheet)
        set_in_progress = DisplaySet(annotator=annotator)
        for row in range(1, sh.nrows): # index by 1 to skip the header row
            if sh.cell_value(rowx=row, colx=0) != "***":
                set_in_progress.sentences.append(sh.cell_value(rowx=row, colx=0))
                set_in_progress.mention_set = sent2mentionset[sh.cell_value(rowx=row, colx=0)]
                if set_in_progress.mention_set is not None:
                    assert set_in_progress.mention_set == sent2mentionset[sh.cell_value(rowx=row, colx=0)]
                if sh.cell_value(rowx=row, colx=1) == "best":
                    set_in_progress.best = sh.cell_value(rowx=row, colx=0)
                if sh.cell_value(rowx=row, colx=1) == "worst":
                    set_in_progress.worst = sh.cell_value(rowx=row, colx=0)
                assert sh.cell_value(rowx=row, colx=1) in ["best", "worst", "", "-"]
            if sh.cell_value(rowx=row, colx=0) == "***":
                set_in_progress.validate()
                out.append(set_in_progress)
                set_in_progress = DisplaySet(annotator=annotator)
    return out


def load_mention_sets_and_display_sets():
    mention_sets_to_display_sets = defaultdict(list)
    for annotator in [1,2,3]:
        display_sets = get_display_sets_from_excel_sheet(file_name="data_from_annotation_firm/{}.xlsx".format(annotator),
                                                         annotator=annotator)
        for display_set in display_sets:
            mention_sets_to_display_sets[display_set.mention_set].append(display_set)
    return mention_sets_to_display_sets    
