# -*- coding: utf-8 -*-
"""
BERT 微调：中文评论情感二分类（简历上"BERT F1"数字的来源）

与基线的区别一句话：
  基线是"数词"，BERT 是"读懂"——预训练已经学会了中文语义，
  微调只是让它学会"你这批评论里什么算好评"。

训练流程（面试必考）：
  1. 分词器把句子切成 token 并映射成 ID（BERT 的输入格式）
  2. 模型前向传播算出"好评/差评"的概率
  3. 用交叉熵损失衡量"预测 vs 真实标签"的差距
  4. 反向传播 + AdamW 更新参数（只微调，lr 很小 2e-5，防止把预训练知识冲掉）
  5. 每个 epoch 结束在测试集上算 F1，存最好的
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

# ---------- 0. 环境与设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备: {device}")
torch.manual_seed(42)  # 固定随机种子，结果可复现

# ---------- 1. 数据 ----------
ds = load_dataset("lansinuote/ChnSentiCorp")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

MAX_LEN = 128  # 每条评论最多取128个token，超出截断（评论文本不长，够用）

class CommentDataset(Dataset):
    """把 HF 数据集包成 PyTorch 能喂给 DataLoader 的格式"""
    def __init__(self, split):
        self.texts = ds[split]["text"]
        self.labels = ds[split]["label"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 分词：截断到 MAX_LEN，不足补 padding（pad 到统一长度才能拼成批次）
        enc = tokenizer(self.texts[idx], truncation=True, max_length=MAX_LEN, padding="max_length")
        return {
            "input_ids": torch.tensor(enc["input_ids"]),        # 文字 -> token ID 序列
            "attention_mask": torch.tensor(enc["attention_mask"]),  # 标记哪些位置是真实内容、哪些是padding
            "labels": torch.tensor(self.labels[idx]),
        }

BATCH_SIZE = 32
train_loader = DataLoader(CommentDataset("train"), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(CommentDataset("test"), batch_size=BATCH_SIZE)

# ---------- 2. 模型 ----------
# 在预训练 BERT 头上加一个二分类层（num_labels=2）
model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
model.to(device)

# ---------- 3. 训练配置 ----------
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5)  # 微调用小学习率，避免破坏预训练知识
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
criterion = torch.nn.CrossEntropyLoss()

best_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    # ---- 训练一轮 ----
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()                 # 清空上一步的梯度
        logits = model(**batch).logits        # 前向传播：输出两个类别的打分
        loss = criterion(logits, batch["labels"])  # 交叉熵损失
        loss.backward()                       # 反向传播算梯度
        optimizer.step()                      # 更新参数
        scheduler.step()                      # 学习率按计划衰减
        total_loss += loss.item()
        if step % 100 == 0:
            print(f"epoch {epoch} | step {step}/{len(train_loader)} | loss {loss.item():.4f}")
    print(f"epoch {epoch} 训练完成，平均 loss: {total_loss/len(train_loader):.4f}")

    # ---- 本轮结束，在测试集上评估（模型从没见过）----
    model.eval()
    preds, trues = [], []
    with torch.no_grad():  # 评估不需要梯度，省显存提速
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(**batch).logits.argmax(dim=-1)  # 取打分最大的类
            preds += pred.cpu().tolist()
            trues += batch["labels"].cpu().tolist()
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    print(f"epoch {epoch} 测试集: Accuracy {acc:.4f} | F1(macro) {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        model.save_pretrained("bert_ckpt")   # 保存最好的模型权重
        tokenizer.save_pretrained("bert_ckpt")
        print(f"  ^ 新最佳，已保存到 bert_ckpt/")

print(f"\n=== BERT 微调最终结果 ===")
print(f"最佳 F1 (macro): {best_f1:.4f}   <- 这是要填进简历的'BERT F1'")
