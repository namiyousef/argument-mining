from argminer.data import DataProcessor
import pandas as pd
def _generate_df_text_from_input(text_segments, strategy):
    text_segments_split = [
        (text_segment.split('::') for text_segment in doc) for doc in text_segments
    ]


    df = pd.concat([
        pd.DataFrame.from_records(
            doc,
            columns=['label', 'text']
        ).assign(doc_id=i) for i, doc in enumerate(text_segments_split)
    ])

    processor = DataProcessor('').from_json(status='preprocessed', df=df).process(strategy).postprocess()
    return processor.dataframe