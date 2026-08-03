import kagglehub
import pandas as pd
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR,"data", "comments.csv" )

def load_data():

    df = pd.read_csv(file_path)
    df = df.dropna(subset=['CommentText', 'Sentiment'])
    df['text'] = df['CommentText']
    df['label'] = df['Sentiment']
    
    return df[['text', 'label']].reset_index(drop=True)

if __name__ == "__main__":
    df = load_data()
    print("data info:", df.info())
    print(df.head())