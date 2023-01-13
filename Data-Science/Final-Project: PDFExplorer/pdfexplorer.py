import os
import fitz
import spacy
import nltk
import shutil
import plotly 


import utils
import time 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from nltk.probability import FreqDist
from nltk.tokenize import wordpunct_tokenize

import pandas as pd
import plotly.express as px


class PDFExplorer(object):
    def __init__(self, path):
        self.path = path
        self.corpus = []
        self.aliases = None
        self.processed_corpus = []

        self.clustered_articles = None

        self.num_clust = None 
        self.cluster_names = None
        self.clustering_done = False

        self.bow_matrix_df = None
        self.bow_matrix = None

        self.cos_sim_scores = None 

    def load_pdfs(self, verbose=False):
        pdf_files = os.listdir(self.path)
        read_in_file_names = []
        if len(pdf_files) < 2:
            raise Exception(
                f"There are less than two files in the specified path:{self.path}"
            )
        for idx, file in enumerate(pdf_files):
            file_path = f"{self.path}/{file}"
            if verbose:
                print(file_path)
            try:
                with fitz.open(file_path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    # Sometimes there could be a scanned pdf file which doesn't lead to an error but yields only an empty text string
                    if text == "":
                        if verbose:
                            print(f"There was a problem reading the file: {file}")
                            print("Skipping...")
                        continue
                    #Some rudimentary cleaning using RegEx
                    text = utils.clean_text(text)
                    self.corpus.append(text)
                    read_in_file_names.append(file)
            except Exception as e:
                if verbose:
                    print(
                        f"The following error occured reading the file {file}: {e}")
                    print("Skipping ...")
                continue
            idx_list = list(range(0, len(self.corpus)))
            self.aliases = {
                idx: name for (idx, name) in list(zip(idx_list, read_in_file_names))
            }

#TODO: How to multithread / run on GPU maybe? 
    def preprocess(self, verbose=False):
        # init spacy
        nlp = spacy.load("en_core_web_lg")
        for idx,raw_text in enumerate(self.corpus):
            doc = nlp(raw_text)
            if verbose:
                article_name = list(self.aliases.values())[idx]
                print("BEFORE PROCESSING:")
                print(f"Article {article_name} has length {len(raw_text)}")
                print("\n")

            # I tried doing all this in one list comprehension, but there was no speed improvement
            # Thus, I leave it like this for better readability

            # Filter out stop words
            filt_doc = [token for token in doc if not token.is_stop]
            # Filter out "words" that are smaller then 3
            filt_doc = [token for token in filt_doc if not len(token) <= 3]
            # Lemmatize (I prefer this to stemming)
            filt_doc = [token.lemma_ for token in filt_doc]

            # Reconstruct the text
            processed_text = " ".join(filt_doc)
            processed_text = processed_text.strip()  # For some trailing whitespaces
            if verbose:
                print("AFTER PROCESSING:")
                print(f"Article {article_name} has length {len(processed_text)}")
                print("\n")

            self.processed_corpus.append(processed_text)

    def create_bow_matrix(self):
        tf_vec = TfidfVectorizer(
            lowercase=True, max_df=0.9, min_df=2, ngram_range=(1, 3)
        )
        bow_matrix = tf_vec.fit_transform(self.processed_corpus)
        self.bow_matrix = bow_matrix
        # Create a dataframe
        df_matrix = pd.DataFrame(
            bow_matrix.toarray(), columns=tf_vec.get_feature_names_out()
        )
        self.bow_matrix_df = df_matrix

    def cluster_articles(self,number_of_clusters=None,verbose=False):
        #If the user doesn't specify the numbers of cluster
        num_articles = len(self.corpus) 

        if number_of_clusters == None and num_articles >= 3: 
            inertia = []
            for k in range(1,num_articles):
                kmeans = KMeans(n_clusters=k).fit(self.cos_sim_scores)
                inertia.append(kmeans.inertia_)
            #Set the number of clusters by the inertia scores 
            inertia_ser = pd.Series(inertia)
            interia_ser_gradient = inertia_ser.diff() / num_articles
            #Now find the biggest change in the gradient and grab the number
            #i.e. find the "ellbow"
            number_of_clusters = interia_ser_gradient.abs().diff().sort_values().index[0]
            if verbose:
                print(f"Decided for {number_of_clusters} Clusters!")

            kmeans = KMeans(number_of_clusters)
            kmeans.fit(self.cos_sim_scores)
        else: 
            kmeans = KMeans(number_of_clusters)
            kmeans.fit(self.cos_sim_scores)

        #Create DataFrame
        df_clusters = pd.DataFrame({"Cluster":kmeans.labels_,"Articles":self.aliases.values()})     
        self.clustered_articles = df_clusters
        self.num_clust = number_of_clusters
        self.clustering_done = True 
        
    def organize_articles(self,copy=True):
        if self.cluster_names == None: 
            self.generate_cluster_names()
        
        for i,repr_bigram_str in enumerate(self.cluster_names):
            #Create a folder 
            dir_name = f"{self.path}/{repr_bigram_str}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            else: 
                #If the repr_bigram_str is not unique 
                dir_name = f"{dir_name}_{i}" 
                os.makedirs(dir_name)
            grouped_articles = list(self.clustered_articles.groupby("Cluster").get_group(i).index)
            for article_id in grouped_articles: 
                article_name = utils.get_article_name(self,article_id)
                article_path = f"{self.path}/{article_name}"
                new_article_path = f"{dir_name}/{article_name}"
                if copy: 
                    shutil.copy(article_path,new_article_path)
                else: 
                    shutil.move(article_path,new_article_path)  
    
    def get_similar_articles(self,article_name): 
        df = self.clustered_articles.set_index("Articles")
        cluster_id = df.loc[article_name,"Cluster"]
        similar_articles = list(df.groupby("Cluster").get_group(cluster_id).index)
        similar_articles.remove(article_name)
        return similar_articles   

    def visualize_articles(self):
        bow_array = self.bow_matrix_df.to_numpy()
        pca = PCA()
        pcs_bow = pca.fit_transform(bow_array)
        if not self.clustering_done: 
            print("Visualizing articles before clustering...")
            time.sleep(2)
            fig = px.scatter(x=pcs_bow[:,0],y=pcs_bow[:,1],
                                hover_name=self.aliases.values(),
                                title="PCA Plot of the Articles",
                                labels={"x":"","y":""})
        else: 
            plot_df = self.generate_plotting_df()
            plot_df["PCA1"] = pcs_bow[:,0]
            plot_df["PCA2"] = pcs_bow[:,1]
            
            print("Visualizing articles after clustering...")
            time.sleep(1)

            
            fig = px.scatter(x=plot_df["PCA1"],y=plot_df["PCA2"],
                    hover_name=plot_df["Article_Names"],
                    color=plot_df["Cluster_Names"],
                    title="PCA Plot of the Articles",
                    labels={"x":"","y":""})
            fig.update_layout(legend_title="Cluster Names")


        #Hide x and y axis 
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        #TODO: REMOVE HOVER_DATA! OR add text sample to hover data 

        #Create Directory for the scatterplots 
        images_dir = f"{self.path}/plots"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        #Create unique filename
        fig_name = f"{images_dir}/scatterplot_{len(os.listdir(images_dir))}.html"
        plotly.offline.plot(fig,filename=fig_name)


    def generate_cluster_names(self):
        repr_bigram_strs = []
        for i in range(0,self.num_clust): 
            grouped_articles = list(self.clustered_articles.groupby("Cluster").get_group(i).index)
            text = ""
            #Create representative bigram for each cluster
            for article in grouped_articles:
                #previously processed corpus
                text += self.processed_corpus[article]
            tokenized_text = wordpunct_tokenize(text)
            bigram = list(nltk.bigrams(tokenized_text))
            repr_bigram = [x for x,_ in FreqDist(bigram).most_common(1)]
            repr_bigram_str = "_".join(repr_bigram[0]) 
            repr_bigram_strs.append(repr_bigram_str)
        self.cluster_names = repr_bigram_strs


    def calculate_cos_sim(self):
        self.cos_sim_scores = cosine_similarity(self.bow_matrix)



    #Custom functions for plotting 
    def set_cluster_names(self): 
        clusters_ser = self.clustered_articles["Cluster"]
        self.generate_cluster_names()
        repl_dic = {k:v for (k,v) in enumerate(self.cluster_names)}
        cluster_names_ser = clusters_ser.replace(repl_dic)
        return cluster_names_ser

    def generate_plotting_df(self):
        cluster_names = self.set_cluster_names()
        article_names = self.clustered_articles["Articles"]
        df_plotting = pd.DataFrame({
                "Cluster_Names":cluster_names,
                "Article_Names":article_names,

        })
        return df_plotting


