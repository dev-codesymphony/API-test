import os, requests, shutil, glove # GloVe
from gensim.models import Word2Vec
import gensim.downloader
from gensim.models import KeyedVectors

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import to_categorical

from django.shortcuts import render
from rest_framework.views import APIView
from django.http import JsonResponse
from rest_framework.response import Response

from rest_framework.permissions import AllowAny
from django.views.generic.base import TemplateView

from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from .serializers import (
    ClassificationGetResponseSerializer,
    ClassificationPostRequestSerializer,
    ClassificationPostSuccessResponseSerializer,
    ClassificationPostErrorResponseSerializer
)

# category labels
main_labels = ['confident', 'unconfident',
               'pos_hp', 'neg_hp',
               'interested', 'uninterested',
               'happy', 'unhappy',
               'friendly', 'unfriendly'
               ]

# to dictionary
label_dict = dict(zip(main_labels, range(0, len(main_labels))))

# inverting label_dict
inv_label = {v: k for k, v in label_dict.items()}

# Hyperparameters
EMBEDDING_DIM = 50

# GloVe
glove_vectors = KeyedVectors.load("glove-twitter-50.kv")

##################################################################################################################################################################


#glove_dir = './data/RNN/'
glove_100k_50d = 'glove.first-100k.6B.50d.txt'
#glove_100k_50d_path = os.path.join(glove_100k_50d)

glove_100k_50d_path = glove_100k_50d

# These are temporary files if we need to download it from the original source (slow)
data_cache = './data/cache'
glove_full_tar = 'glove.6B.zip'
glove_full_50d = 'glove.6B.50d.txt'

#force_download_from_original=False
download_url= 'http://redcatlabs.com/downloads/deep-learning-workshop/notebooks/data/RNN/'+ glove_100k_50d
original_url = 'http://nlp.stanford.edu/data/'+glove_full_tar


#Load glove embeddings
word_embedding = glove.Glove.load_stanford( glove_100k_50d_path )
word_embedding.word_vectors.shape


def get_embedding_vec(word):
    idx = word_embedding.dictionary.get(word.lower(), -1)
    if idx<0:
        return np.zeros(  (EMBEDDING_DIM, ), dtype='float32')  # UNK
    return word_embedding.word_vectors[idx]

def  showChart ( points , position ):
    plt . scatter ( points [:, 0 ], points [:, 1 ], c = position )
    plt . show ()
    return

def  knnCluster (train ,
               num_clusters=5,
               columns=None,
               verbose=0):

    if columns is None:
        columns = ['Valence', 'Arousal','Dominance']

    #print('columns:%s' % columns)
    #print('-' * 40)

    filter_xy = train[columns].values
    random_seed = 1
    kmeans  =  KMeans ( n_clusters = num_clusters , random_state = random_seed )
    class_arr = kmeans.fit_predict(filter_xy)
    if  verbose  ==  1 :
        showChart(train[['Valence', 'Arousal','Dominance']].values, class_arr)

    loss = sum(np.min(cdist(filter_xy, kmeans.cluster_centers_, 'euclidean'), axis=1)) / len(filter_xy)
    train.loc[:, 'Cluster_ID'] = class_arr

    return loss





def get_train_test_data(version='1'):

  if version=='1':
    #data = pd.read_csv('C:\\Users\\Acer\\Documents\\coding\\Upwork\\Sentiment-analysis-API\\RESTapi\\NRC-VAD-Lexicon.csv', encoding='utf-8', sep='\t')
    data = pd.read_csv('RESTapi/NRC-VAD-Lexicon.csv', encoding='utf-8', sep='\t')
    data = data.dropna()
    np.random.seed(1)
    ratio = 0.8
    size  =  int (len(data) * ratio)
    shuffle  =  list(range(len(data)))
    np.random.shuffle(shuffle)

    train = data.iloc[shuffle[:size]]
    test = data.iloc[shuffle[size:]]

    return train, test
  else:
    return null,null

class conf:
  top_word = 3
  top_cluster=3
  num_clusters = 8
  threshold = 0.3
  verbose = 1

def get_default_conf():
    return conf

conf = get_default_conf()

def print_array(arrays, count=0):
    if type(arrays) not in [np.ndarray, list,tuple]:
        print(arrays, end=',')
        return

    print('[', end=' ')
    for  items  in  arrays :
        print_array(items, count + 1)

    if count == 1:
        print(']')
    else:
        print(']', end=' ')


def get_predict_result(word,
                       scores,
                       top=1,
                       threshold=0.,
                       use_emotional_lexicon =True,
                       verbose=0,
                       ):
    scores = scores[:top]
    V  = []
    A = []
    D = []
    sim  = []
    for  index , item  in  enumerate ( scores ):
        if index < 1 or item[3][0] > threshold:
            V.extend ([ i [ 1 ] for  i  in  item [ 2 ]])
            A.extend ([ i [ 2 ] for  i  in  item [ 2 ]])
            D.extend ([ i [ 3 ] for  i  in  item [ 2 ]])
            sim.extend(item[-1])


    sim  =  np . asarray ([ item  if  item  >  0  else  0.1  for  item  in  sim ])
    A = np.asarray(A)
    A_scores = sum(A * sim) / sum(sim)
    V  =  np . asarray ( V )
    V_scores  =  sum ( V  *  sim ) /  sum ( sim )
    D  =  np . asarray ( D )
    D_scores  =  sum ( D  *  sim ) /  sum ( sim )
    return V_scores, A_scores, D_scores


def predict(train_data, word, top_word=3, top_cluster=3,
            threshold=0.3,
            use_emotional_lexicon=True,
            verbose=0):

    cluster_scores = []

    for cluster_id, group in train_data.groupby(by=['Cluster_ID']):
        words_sim = np.asarray([get_word_similarity(word, item) for item in group['Word'].values])
        sorted_index = np.argsort(words_sim)[-1::-1]
        top_words = group[['Word', 'Valence', 'Arousal','Dominance']].values[sorted_index[:top_word]]
        top_words_sim = words_sim[sorted_index[:top_word]]
        score = np.mean(words_sim[sorted_index[:top_word]])
        cluster_scores.append([cluster_id, score, top_words, top_words_sim])

    # Sort by score in ascending order
    sorted_cluster_scores = sorted(cluster_scores, key=lambda x: x[1], reverse=True)

    if  verbose  >  1 :
        print_array(sorted_cluster_scores)

    V, A, D = get_predict_result(word, sorted_cluster_scores, top=top_cluster,
                              threshold=threshold,
                              use_emotional_lexicon=use_emotional_lexicon,
                              verbose=verbose,
                              )

    return V, A, D

def word_prediction(word):

  train_data, test_data = get_train_test_data(version='1')

  knnCluster (train_data ,
               num_clusters = conf.num_clusters,
               columns=['Valence', 'Arousal','Dominance'],
               verbose=0)

  V_pred, A_pred, D_pred = predict(train_data, word, top_word=conf.top_word, top_cluster=conf.top_cluster,
                                  threshold=conf.threshold,
                                  use_emotional_lexicon=False,
                                  verbose=conf.verbose)

  return V_pred, A_pred, D_pred


def predict_class(word,valence,arousal,dominance):
  embedding_features = get_embedding_vec(word)
  feats_without_embedding = [valence, arousal, dominance]
  x_t = np.concatenate((feats_without_embedding,embedding_features)).reshape(1,53)
  #model = load_model('C:\\Users\\Acer\\Documents\\coding\\Upwork\\Sentiment-analysis-API\\RESTapi\\tfjsmodel.h5')
  model = load_model('RESTapi/tfjsmodel.h5')
  prediction = model.predict(x=x_t, verbose = 1)



  return inv_label[np.argmax(prediction)]

def word_sentiment_prediction(word):
  Valence, Arousal, Dominance = word_prediction(word)
  sentiment = predict_class(word, Valence, Arousal, Dominance)

  return sentiment

##################################################################################################################################################################

classificationPostSuccessResponseSerializer = openapi.Response(
    'Success classification', ClassificationPostSuccessResponseSerializer)

classificationPostErrorResponseSerializer = openapi.Response(
    'Error on classification', ClassificationPostErrorResponseSerializer)

classificationGetResponse = openapi.Response(
    'Post request information', ClassificationGetResponseSerializer)


# Classification call class
class Classification(APIView):

    permission_classes = (AllowAny,)

    @swagger_auto_schema(
        operation_description="Get classification request information",
        responses={
            200: classificationGetResponse
        }
    )
    def get(self, request, format=None):
        """API View"""
        return Response({'url': '/classification',
                         'type': 'POST',
                         'data': ['word'],
                         'code': 200,
                         'status': 'Success'
                         })

    @swagger_auto_schema(
        request_body=ClassificationPostRequestSerializer,
        responses={
            200: classificationPostSuccessResponseSerializer,
            400: classificationPostErrorResponseSerializer
        }
    )
    def post(self, request):
        try:
            # word to predict with VAD values
            word = request.data.get('word') or ''

            # calling prediction function
            pred = word_sentiment_prediction(word)
            print('Prediction: {}'.format(pred))

            return JsonResponse({'message': 'Prediction: ' + str(pred), 'code': 200, 'status': 'Success'})
        except Exception as e:
            print('email error:', e)
            return JsonResponse({'message': 'Something went wrong', 'code': 400, 'status': 'Error', 'error': str(e)})


class Home(TemplateView):
    template_name = ('home.html')
