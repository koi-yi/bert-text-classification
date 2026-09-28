# -*- coding: utf-8 -*-
"""
基线模型：TF-IDF + 逻辑回归（简历上"基线"数字的来源）

原理一句话：
  TF-IDF 把每条评论变成"词频权重向量"（哪个词出现得多且独特，权重就高），
  逻辑回归在这堆数字上画一条分界线：好评一边、差评一边。
  它不理解语序和语境，是"数词"的朴素考生。

中文注意点：
  中文没有空格分词，sklearn 默认按空格切词会失效。
  所以用 analyzer='char' + 字符 n-gram(1~2)：
  把"服务很好"拆成 服务/务很/很好 这样的字片段，照样能统计。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 1. 加载数据集（已缓存在本地，不会重复下载）
ds = load_dataset("lansinuote/ChnSentiCorp")
train_texts = ds["train"]["text"]
train_labels = ds["train"]["label"]
test_texts = ds["test"]["text"]
test_labels = ds["test"]["label"]
print(f"训练集 {len(train_texts)} 条，测试集 {len(test_texts)} 条")

# 2. TF-IDF 向量化：只在训练集上"学词表"，防止测试集信息泄漏
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=20000)
X_train = vectorizer.fit_transform(train_texts)  # fit+transform：学习词表并把文本变成向量
X_test = vectorizer.transform(test_texts)        # 只 transform：用同一套词表处理测试集

# 3. 训练逻辑回归
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_labels)

# 4. 在测试集上评估（模型从没见过这些数据）
pred = clf.predict(X_test)
acc = accuracy_score(test_labels, pred)
f1 = f1_score(test_labels, pred, average="macro")  # macro：好评/差评两类F1取平均，不偏科

print(f"\n=== 基线结果（TF-IDF + 逻辑回归）===")
print(f"准确率 Accuracy: {acc:.4f}")
print(f"F1 (macro):      {f1:.4f}   <- 这是要填进简历的'基线F1'")
