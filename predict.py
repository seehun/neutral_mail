import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(
        ), attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

    # ★★★현재경로가 model이 있는 폴더여야함★★★


model1 = torch.load('model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
# state_dict를 불러 온 후, 모델에 저장
model1.load_state_dict(torch.load('state_dict.pt'))

checkpoint = torch.load('all.tar')   # dict 불러오기
model1.load_state_dict(checkpoint['model'])


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 1
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

# 토큰화
# tokenizer = get_tokenizer()
vocab = nlp.vocab.BERTVocab.from_sentencepiece(
    tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize
# tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(
        dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(
        another_test, batch_size=batch_size, num_workers=5)

    model1.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long()
        segment_ids = segment_ids.long()

        valid_length = valid_length
        label = label.long()

        out = model1(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("중립")
            elif np.argmax(logits) == 1:
                test_eval.append("기쁨")
            elif np.argmax(logits) == 2:
                test_eval.append("불안")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔")
            elif np.argmax(logits) == 4:
                test_eval.append("분노")
            elif np.argmax(logits) == 5:
                test_eval.append("당황")
            elif np.argmax(logits) == 6:
                test_eval.append("상처")

            # test_eval.append(np.argmax(logits))

        print(test_eval[0])


predict_sentence = '영화에 나오는 귀신이 너무 무섭네요'

predict(predict_sentence)

# 질문 무한반복하기! 0 입력시 종료
end = 1
while end == 1:
    sentence = input("하고싶은 말을 입력해주세요 : ")
    if sentence == '0':
        break
    predict(sentence)
    print("\n")
