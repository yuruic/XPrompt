from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bert_score import score
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity

def compute_bm25_scores(documents, query):
    count_vectorizer = CountVectorizer()
    doc_term_matrix = count_vectorizer.fit_transform(documents)
    doc_freq = doc_term_matrix.toarray()
    bm25 = BM25Okapi(doc_freq)
    query_freq = count_vectorizer.transform([query]).toarray()[0]
    scores = bm25.get_scores(query_freq)

    return scores

def compute_tfidf_scores(documents, query):
    tfidf_vectorizer = TfidfVectorizer()
    documents_with_query = documents + [query]
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents_with_query)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    return cosine_similarities

def compute_sentence_bert_scores(documents, query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings).flatten()

    return cosine_scores.cpu().tolist()  

def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def sentence_diff_batch(cans, refs, diff_dict, type):
    if type == 'bert':
        P, R, F1 = score(cans, [refs]*len(cans), lang="en", verbose=True)
        diff_dict['bert_p']=P; diff_dict['bert_r']=R; diff_dict['bert_f']=F1
    elif type == 'rouge':
        rouge = Rouge()
        for can in cans:
            scores = rouge.get_scores([can], [refs])[0]
            diff_dict['rouge1_p'].append(scores['rouge-1']['p']); diff_dict['rouge1_r'].append(scores['rouge-1']['r']); diff_dict['rouge1_f'].append(scores['rouge-1']['f'])
            diff_dict['rouge2_p'].append(scores['rouge-2']['p']); diff_dict['rouge2_r'].append(scores['rouge-2']['r']); diff_dict['rouge2_f'].append(scores['rouge-2']['f'])
            diff_dict['rougel_p'].append(scores['rouge-l']['p']); diff_dict['rougel_r'].append(scores['rouge-l']['r']); diff_dict['rougel_f'].append(scores['rouge-l']['f'])
    elif type == 'bleu':
        for can in cans:
            bleu_score = sentence_bleu([refs], can)
            diff_dict['bleu'].append(bleu_score)
    elif type == 'tfidf':
        for can in cans:
            if len(can.split()) == 1:
                vectorizer = TfidfVectorizer(token_pattern=r'.')
            else:
                vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([refs, can])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            diff_dict['tfidf'].append(similarity[0][0])
    elif type == 'sentenceBert':
        similarity = compute_sentence_bert_scores(cans, [refs])
        diff_dict['sentenceBert']=similarity
    
    return diff_dict

def response_sim(response, mask_response, diff_dict, data_type='single'):

    '''
    Calculate the sentence similarity between the original response and the response of masked words
    '''
    if data_type == 'single':
        diff_dict = sentence_diff(mask_response, response, diff_dict, 'sentenceBert')
        diff_dict = sentence_diff(mask_response, response, diff_dict, 'rouge')
        diff_dict = sentence_diff(mask_response, response, diff_dict, 'bleu')
        diff_dict = sentence_diff(mask_response, response, diff_dict, 'tfidf')
    
    elif data_type == 'batch':
        diff_dict = sentence_diff_batch(mask_response, response, diff_dict, 'sentenceBert')
        diff_dict = sentence_diff_batch(mask_response, response, diff_dict, 'rouge')
        diff_dict = sentence_diff_batch(mask_response, response, diff_dict, 'bleu')
        diff_dict = sentence_diff_batch(mask_response, response, diff_dict, 'tfidf')

    
    return diff_dict



