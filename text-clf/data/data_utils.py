import pandas as pd
import sweetviz as sv
import os


def transform_data():
    """
    Transform torch data to data format for this repo
    """
    folder = '../data/yelp_review_full_csv'
    
    for d in ['train.csv', 'test.csv']:
        with open(folder.replace('_csv','')+'.'+d.replace('.csv',''), 'w+') as wf:
            with open(os.path.join(folder, d), 'r') as rf:
                lines = rf.readlines()
                for l in lines:
                    splits = l.split(',', 1)
                    assert len(splits) == 2
                    wf.write('__label__' \
                                + splits[0].replace('"','') \
                                + ' , ')
                    wf.write(splits[1].replace('"', ''))


def vis(fn):
    """ 
    Load file to dataframe and dump HTML report
    """
    labels = []
    with open(fn, 'r') as rf:
        lines = rf.readlines()
        for l in lines:
            splits = l.split(' , ', 1)
            assert len(splits) == 2
            labels.append(splits[0])
    df = pd.DataFrame({'label':labels})
    my_report = sv.analyze(df)
    my_report.show_html()


if __name__ == '__main__':
    #vis('yahoo_answers.train.sample')
    transform_data()
