from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

def compute_bm25_scores(documents, query):
    # 使用CountVectorizer来转换文档为词频矩阵
    count_vectorizer = CountVectorizer()
    doc_term_matrix = count_vectorizer.fit_transform(documents)

    # 获得词频矩阵的数组形式
    doc_freq = doc_term_matrix.toarray()

    # 初始化BM25模型
    bm25 = BM25Okapi(doc_freq)

    # 使用训练好的分词器处理查询
    query_freq = count_vectorizer.transform([query]).toarray()[0]

    # 计算每个文档对于查询的BM25分数
    scores = bm25.get_scores(query_freq)

    return scores


def compute_tfidf_scores(documents, query):
    # 初始化TF-IDF向量化器
    tfidf_vectorizer = TfidfVectorizer()

    # 将查询添加到文档集中
    documents_with_query = documents + [query]

    # 训练向量化器并转换文档集为TF-IDF矩阵
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents_with_query)

    # 计算查询向量（最后一个文档）与所有其他文档的余弦相似度
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    return cosine_similarities



def compute_sentence_bert_scores(documents, query):
    # 加载预训练的Sentence-BERT模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 使用模型为文档和查询生成嵌入
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 计算查询与每个文档的余弦相似度
    cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings).flatten()

    return cosine_scores.cpu().tolist()  # 将得分转换为Python列表




