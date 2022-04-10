# -- public imports
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
from torch.utils.data import DataLoader
import torch
from pandas.testing import assert_frame_equal
import time
import warnings
warnings.filterwarnings("ignore")
# -- private imports
##from colabtools.utils import move_to_device

# -- dev imports
#%load_ext autoreload
#%autoreload 2

from argminer.data import ArgumentMiningDataset, TUDarmstadtProcessor, PersuadeProcessor, DataProcessor, create_labels_doc_level, df_from_text_files
from argminer.evaluation import inference


#NEW VERSION
class PersuadeProcessor(DataProcessor):
    def __init__(self,path=''):
        super().__init__(path)
        
    def _preprocess(self):
        path_to_text_dir = os.path.join(self.path, 'train')
        path_to_ground_truth = os.path.join(self.path, 'train.csv')
        
        df_ground_truth = pd.read_csv(path_to_ground_truth)
        df_texts = df_from_text_files(path_to_text_dir)
        
        
        df_ground_truth = df_ground_truth.sort_values(['id', 'discourse_start', 'discourse_end'])
        df_ground_truth = df_ground_truth.drop(columns=['discourse_id','discourse_start','discourse_end','discourse_type_num'])
        
        
        df_texts['text_split'] = df_texts.text.str.split()
        df_texts['range'] = df_texts['text_split'].apply(lambda x: list(range(len(x))))
        df_texts['start_id'] = df_texts['range'].apply(lambda x: x[0])
        df_texts['end_id'] = df_texts['range'].apply(lambda x: x[-1])
        df_texts = df_texts.drop(columns=['text_split','range'])
        
        
        df = df_ground_truth.groupby('id').agg({
            'discourse_text':lambda x: ' '.join(x),
            'predictionstring': lambda x: ' '.join(x),
    
        }).reset_index()
        
        df = df.merge(df_texts)
        
        df['predictionstring'] = df.predictionstring.apply(lambda x: [int(num) for num in x.split()])
        df['pred_str_start_id'] = df.predictionstring.apply(lambda x: x[0])
        df['pred_str_end_id'] =  df.predictionstring.apply(lambda x: x[-1])
        
        new_df = pd.DataFrame()
        for row in df.sort_values('id').itertuples(index=False):
            if row.end_id != row.pred_str_end_id:
                s = row.text.split()[row.pred_str_end_id:]
                new_string = ' '.join(s)
                new_predsStr = list(range(row.pred_str_end_id+1,row.end_id+1))
                new_predsStr = " ".join([str(i) for i in new_predsStr])
                new_type = 'Other'
                new_id = row.id
                new_row = {'id': new_id, 'discourse_text':new_string ,'discourse_type':'Other','predictionstring':new_predsStr}
                new_df = new_df.append(new_row,ignore_index=True)
                
        df_combined = df_ground_truth.append(new_df,ignore_index=True)
        
        ##KEEP THE FOLLOWING OR NOT???
        df_combined.predictionstring = df_combined.predictionstring.apply(lambda x: [int(num) for num in x.split()])
        df_combined['start'] = df_combined.predictionstring.apply(lambda x: x[0])
        df_combined['end'] =  df_combined.predictionstring.apply(lambda x: x[-1])
        df_combined= df_combined.sort_values(['id', 'start', 'end'])

        
        self.dataframe = df_combined
        self.status = 'preprocessed'
        return self
    
    def _process(self, strategy, processors=[]):
        # processes data to standardised format, adds any extra cleaning steps
        assert strategy in {'io', 'bio', 'bieo'} # for now

        df = self.dataframe.copy()

        for processor in processors:
            df['discourse_text'] = df['discourse_text'].apply(processor)


        # add labelling strategy
        label_strat = dict(
            add_end='e' in strategy,
            add_beg='b' in strategy
        )
        
        df['discourse_type'] = df[['discourse_type', 'predictionstring']].apply(
            lambda x: _generate_entity_labels(
                len(x['predictionstring']), x['discourse_type'], **label_strat
            ), axis=1
        )

        self.dataframe = df
        self.status = 'processed'


        return self
    
    def _postprocess(self):
        df_post = self.dataframe.copy()
        df_post['predictionstring'] = df_post['predictionstring'].apply(
                    lambda x: ' '.join([str(item) for item in x])
                )

        df_post['discourse_type'] = df_post['discourse_type'].apply(
                    lambda x: ' '.join([str(item) for item in x])
                )
        df_post = df_post.groupby('id').agg({
            'discourse_text':lambda x: ' '.join(x),
            'predictionstring': lambda x: ' '.join(x),
            'discourse_type': lambda x: ' '.join(x)
        })
        df_post['discourse_type'] = df_post['discourse_type'].str.split()
        df_post['predictionstring'] = df_post['predictionstring'].str.split().apply(lambda x: [int(x) for x in x])

        df_post = df_post.reset_index()
        
        df_post = df_post.rename(columns={'discourse_type':'labels','predictionstring':'predictionString',
                                          'discourse_text':'text','id':'doc_id'})
        self.dataframe = df
        self.status = 'postprocessed'

        return self


def _generate_entity_labels(length, label, add_end=False, add_beg=True):
    """
    For cases where argument segment is only 1 word long, beginning given preference over end
    """
    labels = [f'I-{label}'] if label != 'Other' else ['O']
    labels *= length

    if add_end:
        if label != 'Other':
            labels[-1] = f'E-{label}'

    if add_beg:
        if label != 'Other':
            labels[0] = f'B-{label}'

    return labels

