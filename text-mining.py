import re
import spacy
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from skillcodes import tags_table
from categories import categories_table
from skill_keywords import lookup_table
from numpy import array

en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words
lem = WordNetLemmatizer()
sw_nltk = set(stopwords.words("english"))
dummy_string: str = "Some dummy string to prevent problem of empty vocabulary with sci kit feature extraction"
new_stop_words = (
    "part", 
    "class", 
    "course", 
    "one", 
    "two", 
    "three",
    "four", 
    "discussion", 
    "lecture", 
    "hour", 
    "day", 
    "month", 
    "semester", 
    "week", 
    "sophomore", 
    "junior", 
    "senior", 
    "fresh", 
    "seminar",
    "exam",
    "required",
    "summer",
    "winter",
    "student",
    "pre",
    "requisite",
    "prerequisite",
    "lecture",
    "introduction",
    "introduces",
    "essay",
    "notes",
    "textbook",
    "etc",
    "covering",
    "sp",
    "credit",
    "pr",
    "fsp",
    "info",
    "session",
    "read",
    "basic",
    "hard",
    "emphasis",
    "form",
    "primary",
    "understand",
    "learn",
    "discus",
    "learning",
    "general",
    "concept",
    "study",
    "overciew",
    "focus",
    "emphasize",
    "presented",
    "learning",
    "seminar",
    "proseminar",
    "topic",
    "major",
    "year",
    "distinguished"
    )
sw_nltk = sw_nltk.union(new_stop_words)

def split_text(string: str):
    return str.split('')

def preprocess_tag(tag: str) -> str:
    text = tag
    text = text.lower()
    text = re.sub(r'(\\d|\\W)+',' ',text)
    text = re.sub(r'\((.*?)\)', '', text)
    text = re.sub(r'[()]', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', ' ', text)

    return text

def preprocess_tags(corpus: list) -> list:
    length_of_corpus: int = len(corpus)
    processed_corpus: list = []
    text: str = ''

    if length_of_corpus == 0:
        return

    for document in corpus:
        text = document
        text = text.lower()
        text = re.sub(r'(\\d|\\W)+',' ',text)
        text = re.sub(r'\((.*?)\)', '', text)
        text = re.sub(r'[()]', ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', ' ', text)
        processed_corpus.append(text)
    return processed_corpus        

def preprocessing(corpus: list) -> list:
    length_of_corpus: int = len(corpus)
    processed_corpus: list = []
    processed_text: str = ''
    text: str = ''

    if length_of_corpus == 0:
        return

    for document in corpus:
        text = document
        text = text.lower()
        text = re.sub('(\\d|\\W)+',' ',text)
        text = re.sub('[()]', ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = word_tokenize(text)
        text = [word for word in text if word not in sw_spacy]
        # text = [word for word in text if word not in sw_nltk]
        text = [lem.lemmatize(word) for word in text]
        text = [word for word in text if word not in sw_nltk]
        processed_text = ' '.join(text)

        try:
            keyword_vector = CountVectorizer(
                encoding='latin-1'
                ).fit([processed_text])
        except ValueError:
            keyword_vector = CountVectorizer(
                 encoding='latin-1'
                ).fit([dummy_string])    

        extracted_keywords = [word for word, idx in keyword_vector.vocabulary_.items()]
        processed_corpus.append(extracted_keywords)        

    return processed_corpus

def preprocessing_test_cases(corpus: str):
    length_of_corpus: int = len(corpus)
    processed_text: str = ''
    text: str = ''

    if length_of_corpus == 0:
        return

    
    text = corpus
    text = text.lower()
    text = re.sub('(\\d|\\W)+',' ',text)
    text = re.sub('[()]', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in sw_spacy]
    # text = [word for word in text if word not in sw_nltk]
    text = [lem.lemmatize(word) for word in text]
    text = [word for word in text if word not in sw_nltk]
    processed_text = ' '.join(text)

    try:
        keyword_vector = CountVectorizer(
            encoding='latin-1'
            ).fit([processed_text])
    except ValueError:
        keyword_vector = CountVectorizer(
            encoding='latin-1'
            ).fit([dummy_string])    

    extracted_keywords = [word for word, idx in keyword_vector.vocabulary_.items()] 
    #return [' '.join(extracted_keywords)]
    return extracted_keywords   

def create_lookup_table():
    data_frame: DataFrame = pd.read_csv(r'./Courses_Berkeley_2018-01-15.csv', encoding='latin1')
    data_frame = data_frame.dropna(subset=["Description"])
    data_frame = data_frame[data_frame['Description'].map(len) > 10]
    #data_frame = data_frame.drop_duplicates(subset=['Description'], keep='first', inplace=True)
    #data_frame = data_frame[data_frame['Name'].isin(['seminar', 'lectures', 'discussions', 'group', 'study', 'studies', 'course', 'courses', 'quarter' ])]
    #data_frame = data_frame[0:2000]
    
    data_frame_length: int = len(data_frame)

    course_description: Series = data_frame['Description']
    course_name: Series = data_frame['Name']

    course_description_array: list = []
    for i in range(data_frame_length):
        course_description_array.append(course_description.values[i])

    course_names_array: list = []
    for i in range(data_frame_length):
        course_names_array.append(course_name.values[i])    

    processed_course_description = preprocessing(course_description_array)
    processed_course_tags = preprocess_tags(course_names_array)

    file = open('keywords.py', "a", encoding="utf-8")
    file.write('lookup_table = {')

    for i in range(data_frame_length):
        file.write(f'\n\t\'{tags_table[processed_course_tags[i]]}\' : {processed_course_description[i]},')
    file.write('\n}')
    file.close()

def create_tags_table():
    skill_codes_initial: int = 10000
    data_frame: DataFrame = pd.read_csv(
        r'Courses_Berkeley_2018-01-15.csv', 
        encoding='latin1'
        ).dropna(subset=["Description"])
    #data_frame = data_frame[~data_frame['Name'].map(split_text).map().isin(['seminar', 'lectures', 'discussions', 'group', 'study', 'studies', 'course', 'courses', 'quarter' ])]
    unique_keys: list = data_frame['Name'].unique()
    processed_unique_keys: list = preprocess_tags(unique_keys)

    file = open('skillcodes.py', "a", encoding="utf-8")
    file.write('tags_table = {')
    for key in processed_unique_keys:
        file.write(f'\n\t\'{key}\' : {skill_codes_initial},')
        skill_codes_initial = skill_codes_initial + 1
    file.write('\n}')
    file.close()

def create_categories_table():
    data_frame: DataFrame = pd.read_csv(r'./Courses_Berkeley_2018-01-15.csv', encoding='latin1')
    data_frame = data_frame.dropna(subset=["Description"])
    data_frame = data_frame[data_frame['Description'].map(len) > 10]

    unique_categories: list = data_frame['Field'].unique()

    for key in unique_categories:
        print(key)
    print(len(unique_categories))

    category_code: int = 1

    file = open('categories.py', "a", encoding="utf-8")
    file.write('categories_table = {')
    for key in unique_categories:
        file.write(f'\n\t\'{key}\' : {category_code},')
        category_code = category_code + 1
    file.write('\n}')
    file.close()

def create_category_skill_mapping():
    data_frame: DataFrame = pd.read_csv(r'./Courses_Berkeley_2018-01-15.csv', encoding='latin1')
    data_frame = data_frame.dropna(subset=["Description"])
    data_frame = data_frame[data_frame['Description'].map(len) > 10]
    data_frame.drop(['Year'], axis=1, inplace=True)
    data_frame.drop(['Number'], axis=1, inplace=True)
    data_frame.drop(['Winter'], axis=1, inplace=True)
    data_frame.drop(['Spring'], axis=1, inplace=True)
    data_frame.drop(['Summer'], axis=1, inplace=True)
    data_frame.drop(['Taught'], axis=1, inplace=True)
    data_frame.drop(['Profs1'], axis=1, inplace=True)
    data_frame.drop(['Profs2'], axis=1, inplace=True)
    data_frame.drop(['Fall'], axis=1, inplace=True)
    data_frame.drop(['GenArea'], axis=1, inplace=True)

    sheet_number:int = 1
    mapping_data_frame: DataFrame = pd.DataFrame(columns=['Field','Name', 'Area', 'Description'])

    writer = pd.ExcelWriter('Categories_Data.xlsx', engine = 'xlsxwriter')

    for key in categories_table:
        df = data_frame.loc[data_frame['Field'] == key]
        df.drop_duplicates(subset="Name", keep=False, inplace=True)
        df = df.loc[df['Name'].map(preprocess_tag).isin(tags_table.keys())]
        #mapping_data_frame.append(df)
        df.to_excel(writer, sheet_name = f'{sheet_number}')
        sheet_number = sheet_number + 1 

    writer.save()
    writer.close()
     

    #mapping_data_frame.to_excel('Categories_Data.xlsx', f{sheet_number})

def create_single_label_dataset():
    writer = pd.ExcelWriter('single_label_data.xlsx', engine='xlsxwriter')
    count: int = 1
    df: DataFrame = pd.DataFrame()

    df['docs'] = ''
    df['labels'] = ''

    for key in lookup_table:
        array = [' '.join(lookup_table[key]), key]
        df.loc[count] = array
        count = count + 1

    print(df)   
    df.to_excel('single_label_data.xlsx')



def test():
    data_frame: DataFrame = pd.read_csv(r'./Courses_Berkeley_2018-01-15.csv', encoding='latin1')
    data_frame = data_frame.dropna(subset=["Description"])
    data_frame = data_frame[data_frame['Description'].map(len) > 10]
    data_frame = data_frame[0:2000]
    
    data_frame_length: int = len(data_frame)

    course_name: Series = data_frame['Name']

    course_names_array: list = []
    for i in range(data_frame_length):
        course_names_array.append(course_name.values[i])    

    processed_course_array = preprocess_tags(course_names_array)

    for i in range(2000):
        print(f'{course_names_array[i]} --> {processed_course_array[i]}')


def main():
    #create_lookup_table()
    #test()
    #create_tags_table()
    #preprocessing_test_cases("yaba daba dododo")
    #create_categories_table()
    #create_category_skill_mapping()
    create_single_label_dataset()

if __name__ == "__main__":
    main()    