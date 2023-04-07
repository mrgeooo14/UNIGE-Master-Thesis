import pandas as pd
import json

#### Converting a Pandas Dataframe to a JSON Lines format
#### This is the format that Prodigy prefers to deal with NER annotation
def convert_string_to_json(string):
    text = ' '.join([str(c).replace('\n', ' ') for c in string])
    json_dict = {"text": text}
    json_str = json.dumps(json_dict)
    return json_str

def dataframe_row_to_jsonl(dataframe, savefile = True):
    json_lines = []
    for i, row in dataframe.iterrows():
        json_str = convert_string_to_json(row)
        json_lines.append(json_str)
    
    if savefile:
        with open('dataset_for_prodigy.jsonl', 'w') as outfile:
            outfile.write('\n'.join(json_lines))
            
#### Our sentences were stored in the first row of an excel file
if __name__ == "__main__":
    df = pd.read_excel('data/rondine.xlsx', usecols = [0])
    dataframe_row_to_jsonl(df)
    print('Excel data successfully converted to a JSON Liner format')
