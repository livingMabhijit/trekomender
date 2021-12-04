from flask import Flask,render_template,request
import time
import pandas as pd
from rank_bm25 import BM25Okapi
import tqdm
import spacy
from rank_bm25 import BM25Okapi
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")
import annoy
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/bestMatch', methods = ['GET','POST'])
def best_match():
    search_string = request.args.get('q')
    ## start ##
    df = pd.read_csv('City_data1.csv')
    df['out'] = df['sights']+df['desc']

    text_list = df.desc.str.lower().values
    tok_text=[] # for our tokenised corpus
    #Tokenising using SpaCy:
    for doc in tqdm(nlp.pipe(text_list, disable=["tagger", "parser","ner"])):
        tok = [t.text for t in doc if t.is_alpha]
        tok_text.append(tok)
    bm25 = BM25Okapi(tok_text)
    tokenized_query = str(search_string).lower().split(" ")
  
    t0 = time.time()
    results = bm25.get_top_n(tokenized_query, df.desc.values, n=6)
    t1 = time.time()
    print(f'Searched 50,000 records in {round(t1-t0,3) } seconds \n')

    cities = []
    countries = []
    imgs = []
    for i in results:
        city = df[df['desc']==str(i)]['city'].iloc[0]
        cities.append(city)
        cont = df[df['desc']==str(i)]['country'].iloc[0]
        countries.append(cont)
        img = df[df['desc']==str(i)]['img'].iloc[0]
        imgs.append(str(img))
    ## end ##
    
    return render_template('best_match.html',results = results,cities = cities,countries = countries, imgs=imgs)


@app.route('/sim_city', methods = ['GET','POST'])
def similar_cities():
    df = pd.read_csv('City_data1.csv')
    user_inputs = []
    
    input1 =  request.args.get('place1')
    # input2 =  request.args.get('q2')
    # input3 =  request.args.get('q3')

    df1 = df[['city','beach','mountain','history','food','city_life','countryside','nightlife','couple_friendly','outdoor','spiritual']]
    df2 = df1.set_index('city').T.to_dict('list')
    data = dict()
    names = []
    vectors = []

    for a,b in df2.items():
        names.append(a)
        vectors.append(b)

    data['name'] = np.array(names, dtype=object)
    data['vector'] = np.array(vectors, dtype=float)

    class AnnoyIndex():
        def __init__(self, vector, labels):
            self.dimention = vector.shape[1]
            self.vector = vector.astype('float32')
            self.labels = labels


        def build(self, number_of_trees=5):
            self.index = annoy.AnnoyIndex(self.dimention)
            for i, vec in enumerate(self.vector):
                self.index.add_item(i, vec.tolist())
            self.index.build(number_of_trees)
            
        def query(self, vector, k=10):
            indices = self.index.get_nns_by_vector(vector.tolist(), k)
            return [self.labels[i] for i in indices]
    index = AnnoyIndex(data["vector"], data["name"])
    index.build()

    place_vector, place_name = data['vector'][list(data['name']).index(str(input1))], data['name'][list(data['name']).index(str(input1))]
    start = time.time()
    simlar_place_names = index.query(place_vector)
    print(f"The most similar places to {place_name} are {simlar_place_names}")
    # print(simlar_place_names)
    end = time.time()
    print(end - start)
    # simlar_place_names = [place for place in simlar_place_names]
    # print(simlar_place_names)
    my_dest = str(simlar_place_names)
    cities = []
    countries = []
    imgs = []
    descrs = []
    for i in simlar_place_names:
        # print(i)
        city = df[df['city']==str(i)]['city'].iloc[0]
        cities.append(city)
        cont = df[df['city']==str(i)]['country'].iloc[0]
        countries.append(cont)
        img = df[df['city']==str(i)]['img'].iloc[0]
        imgs.append(str(img))
        descr = df[df['city']==str(i)]['desc'].iloc[0]
        descrs.append(descr)
    
    
    
    
    # print(simlar_place_names.split('*'))
    return render_template('sim_city.html',place_name = place_name,cities = cities,countries = countries, imgs=imgs,descrs = descrs)#,countries = countries, imgs=imgs)simlar_place_names = simlar_place_names,
    # return cities




@app.route('/about', methods = ['GET'])
def abouts():
    return render_template('about.html')

@app.route('/sights', methods = ['GET'])
def sights():




    return city#render_template('sights.html')


if __name__ == "__main__":
    app.run(debug = True)
