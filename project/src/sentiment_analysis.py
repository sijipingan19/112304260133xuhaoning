import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle

# 下载nltk资源
nltk.download('punkt')
nltk.download('stopwords')

# 数据清洗函数
def clean_text(text):
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 移除标点符号和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 转为小写
    text = text.lower()
    # 分词
    tokens = word_tokenize(text)
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # 过滤空字符串
    tokens = [word for word in tokens if word.strip()]
    return tokens

# 加载数据
print("加载数据...")
train_data = pd.read_csv('labeledTrainData.tsv', sep='\t', quoting=3)
test_data = pd.read_csv('testData.tsv', sep='\t', quoting=3)

# 清洗训练数据
print("清洗训练数据...")
train_data['cleaned_review'] = train_data['review'].apply(clean_text)

# 清洗测试数据
print("清洗测试数据...")
test_data['cleaned_review'] = test_data['review'].apply(clean_text)

# 训练Word2Vec模型
print("训练Word2Vec模型...")
all_reviews = train_data['cleaned_review'].tolist() + test_data['cleaned_review'].tolist()
word2vec_model = Word2Vec(sentences=all_reviews, vector_size=300, window=10, min_count=3, workers=4)

# 保存Word2Vec模型
print("保存Word2Vec模型...")
word2vec_model.save('word2vec.model')

# 计算文档向量（均值embedding）
def get_document_vector(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# 生成训练数据的向量
print("生成训练数据向量...")
train_vectors = np.array([get_document_vector(tokens, word2vec_model) for tokens in train_data['cleaned_review']])
y_train = train_data['sentiment'].values

# 生成测试数据的向量
print("生成测试数据向量...")
test_vectors = np.array([get_document_vector(tokens, word2vec_model) for tokens in test_data['cleaned_review']])

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_vectors, y_train, test_size=0.2, random_state=42)

# 训练逻辑回归模型
print("训练逻辑回归模型...")
lr_model = LogisticRegression(max_iter=1000, C=10.0, random_state=42)
lr_model.fit(X_train, y_train)

# 在验证集上评估
print("在验证集上评估...")
y_val_pred = lr_model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_val_pred)
print(f"验证集AUC分数: {auc_score}")

# 保存逻辑回归模型
print("保存逻辑回归模型...")
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# 对测试数据进行预测
print("对测试数据进行预测...")
test_pred = lr_model.predict_proba(test_vectors)[:, 1]

# 生成提交文件
print("生成提交文件...")
# 移除ID列中的多余引号
submission = pd.DataFrame({'id': test_data['id'].str.strip('"'), 'sentiment': test_pred})
submission.to_csv('submission.csv', index=False, quoting=1)

print("任务完成！")
