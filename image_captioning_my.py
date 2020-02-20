import torch, torch.nn as nn
import torch.nn.functional as F
import time
import math
import random

import torchvision
from torchvision.models.inception import Inception3
from warnings import warn

import numpy as np
import json

UNK, BOS, EOS, PAD  = "UNK", "BOS", "EOS", "PAD"
UNK_IX, BOS_IX, EOS_IX, PAD_IX = 0, 1, 2, 3


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BeheadedInception3(Inception3):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """
    
    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        else: warn("Input isn't transformed")
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x_for_capt = x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        return x_for_attn, x_for_capt, x
    

def beheaded_inception_v3(transform_input=True):
    model= BeheadedInception3(transform_input=transform_input)
    inception_fname = 'saved_models/' + 'inception_v3_google-1a9a5a14.pth'

    state_dict = torch.load(inception_fname)
    model.load_state_dict(state_dict)

    return model


def as_matrix(sequences, tokens, max_len=None):
    """
    Convert a list of tokens into a matrix with padding
    params:
        sequences: list of sentences. Each sentence is a tokenized string or list of tokens
        indices_not_words  =True  - если возвращается матрица индексов слов из исходного предложения (если используем непредобученный слой nn.Embedding)
                           =False - если возвращается матрица слов из исходного предложения (если используем предобученные Embedding-и)
        max_len: if specified,
    """
    token_to_id = {t: i for i, t in enumerate(tokens)}
    if isinstance(sequences[0],
                  str):  # проверка принадлежности данных определенному классу (типу данных str - в данном случае)
        sequences = list(map(str.split, sequences))

    max_len = min(max(map(len, sequences)), max_len or float('inf'))
    # print('max_len =', max_len)

    # max_len = мин из(максимум из длин предложений в sequences; входной max_len (если он есть) или бесконечное значение (если нет  max_len))
    # т.е. если есть входной max_len, то max_len = мин из(максимум из длин предложений в sequences, входной max_len)
    # если нет входного max_len, то  max_len = максимум из длин предложений в sequences

    matrix = np.full((len(sequences), max_len), np.int32(
        PAD_IX))  # матрица размера: количество входных предложений х max_len (сначала заполняем индексом PAD-а)

    for i, seq in enumerate(sequences):  # перенумерованные входные предложения
        row_ix = [token_to_id.get(word, UNK_IX) for word in
                  seq[:max_len]]  # row_ix - это массив индексов слов в очередном предложении
        matrix[i,:len(row_ix)] = row_ix  # i-й строке матрицы (элементам i-й строки от начала до len(row_ix) (:len(row_ix))
        # присвоить row_ix (массив индексов слов в очередном предложении)

    # print ('matrix=\r\n',  matrix[0:6]) # выведем строки matrix с 0-й по 5-ю
    return matrix


class CaptionNet(nn.Module):
    def __init__(self, n_tokens, emb_size=128, lstm_units=256,
                 cnn_feature_size=2048, lstm_num_layers=1, lstm_num=1, lstm_dropout=0, embed_usepretrained=False,  embed_weights_matrix = None):
        super(self.__class__, self).__init__()
        """
        :param: n_tokens - размер словаря
        :param: emb_size is the dimensionality of the embedding layer. 
        :param: lstm_units is the dimensionality of the hidden and cell states.
        :param: cnn_feature_size - длина вектора, получившегося из image при пропускании через сеть-Энкодер
        :param: lstm_num_layers - число слоев в LSTM
        :param: lstm_num  - количество конкатенированных LSTM
        :param: lstm_dropout - параметр dropout для LSTM
        :param: embed_usepretrained  - использовать ли предобученные эмбеддинги (по умолчанию True)
        :param: embed_weights_matrix - матрица весов (предобученных) для слоя nn.Embedding (используется если embed_usepretrained=True)
        """        
        # два линейных слоя, которые будут из векторов, полученных на выходе Inseption, 
        # получать начальные состояния h0 и c0 LSTM-ки, которую мы потом будем 
        # разворачивать во времени и генерить ею текст
        
        self.cnn_to_h = nn.Linear(cnn_feature_size, lstm_units)
        self.cnn_to_c = nn.Linear(cnn_feature_size, lstm_units)
        
        self.lstm_num_layers = lstm_num_layers
        self.lstm_units = lstm_units
        self.lstm_num = lstm_num  # количество конкатенированных LSTM
               
        # вот теперь recurrent part

        # create embedding for input words. Use the parameters (e.g. emb_size).
        #<YOUR CODE>
        if embed_usepretrained and not (embed_weights_matrix is None): #используем предтренированные эмбеддинги #Ex3
            self.embedding_layer = nn.Embedding.from_pretrained(embed_weights_matrix, freeze=False, padding_idx=PAD_IX)   
            print('CaptionNet: embed_usepretrained=True')
        else: # тренируем с нуля (не используем предтренированные эмбеддинги)
            self.embedding_layer = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_size, padding_idx=PAD_IX)
            print('CaptionNet: embed_usepretrained=False')
        
        # lstm: настакайте LSTM-ок (1 или более, но не надо сразу пихать больше двух, замечаетесь ждать).
        #<YOUR CODE>
        if (lstm_num != 1) and (lstm_num != 2): assert 'lstm_num must be 1 or 2 '
        self.lstm = nn.LSTM(input_size = emb_size, 
                            hidden_size = lstm_units,
                            num_layers = lstm_num_layers, 
                            batch_first = True, 
                            dropout=lstm_dropout)
        if (lstm_num == 2):
            self.lstm2 = nn.LSTM(
                # input_size = lstm_units,
                input_size=emb_size,
                hidden_size=lstm_units,
                num_layers=lstm_num_layers,
                batch_first=True,
                dropout=lstm_dropout)
            # ну и линейный слой для получения логитов
        #<YOUR CODE>
        
        self.linear = nn.Linear(lstm_units*lstm_num, n_tokens)
        
    def forward(self, image_vectors, captions_ix, device='cuda'):
        """ 
        Apply the network in training mode. 
        :param image_vectors: torch tensor, содержаший выходы inseption. Те, из которых будем генерить текст
                shape: [batch, cnn_feature_size]
        :param captions_ix: 
                таргет описания картинок в виде матрицы
        :param: device  - 'cuda' if torch.cuda.is_available() else 'cpu'                 
        :returns: логиты для сгенерированного текста описания, shape: [batch, word_i, n_tokens]
        """

        image_vectors_short = self.cnn_to_h(image_vectors)
        image_vectors_short2 = self.cnn_to_h(image_vectors)

        batch_size = captions_ix.shape[0]
        seq_len = captions_ix.shape[1]

        initial_hid_cell = self.init_hidden(batch_size)
        if self.lstm_num_layers == 2:
            initial_hid_cell[0][0] = image_vectors_short  #hidden
            initial_hid_cell[0][1] = image_vectors_short  #hidden
            initial_hid_cell[1][0] = image_vectors_short2  #cell
            initial_hid_cell[1][1] = image_vectors_short2  #cell
        elif self.lstm_num_layers == 1:
            initial_hid_cell[0][0] = image_vectors_short  #hidden
            initial_hid_cell[1][0] = image_vectors_short2  #cell
        else:
            assert 'lstm_num_layers must be 1 or 2'
            
        #captions_emb = <YOUR CODE>
        captions_emb = self.embedding_layer(captions_ix)
        #print('captions_emb.shape=', captions_emb.shape)
        # применим LSTM:
        # 1. инициализируем lstm state с помощью initial_* (сверху)
        # 2. скормим LSTM captions_emb
        # 3. посчитаем логиты из выхода LSTM
       
        #lstm_out = <YOUR_CODE> # shape: [batch, caption_length, lstm_units]
        lstm_out, hid_cell = self.lstm(captions_emb, initial_hid_cell)
        if (self.lstm_num == 2):
            lstm_out2, __ = self.lstm2(captions_emb, initial_hid_cell)
            lstm_out = torch.cat([lstm_out, lstm_out2], dim = 2)

         #logits = <YOUR_CODE> 
        logits = self.linear(lstm_out)

        return logits

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_num_layers, batch_size, self.lstm_units).zero_(), weight.new(self.lstm_num_layers, batch_size, self.lstm_units).zero_()) #TTT comment
        return hidden

def generate_caption(network, image, vocab, caption_prefix=(BOS,),
                     t=1, sample=True, max_len=100, device='cuda'):
    print('generate_caption')
    assert torch.is_tensor(image) and torch.max(image) <= 1.0 and torch.min(image) >=0.0 and image.shape[0] == 3
    inception = beheaded_inception_v3().train(False)
    with torch.no_grad():
        _, vectors_neck, _ = inception(image[None])

        caption_prefix = list(caption_prefix)
        for i in range(max_len):
            # представить в виде матрицы
            #prefix_ix = <YOUR CODE>
            prefix_ix = as_matrix(caption_prefix, vocab, max_len)
            #print('prefix_ix=', prefix_ix)
            prefix_ix = torch.tensor(prefix_ix, dtype=torch.int64).to(device)
            
            #vectors_neck = torch.tensor(vectors_neck, dtype=torch.float32).to(device)
            vectors_neck = vectors_neck.clone().detach().to(device)
            # получаем логиты из RNN-ки

            next_word_logits = network.forward(vectors_neck, prefix_ix, device=device)[0, -1]
             # переводим их в вероятности
            next_word_probs = F.softmax(next_word_logits, dim=-1).data
            next_word_probs = next_word_probs.cpu().clone().detach().numpy().squeeze()
            assert len(next_word_probs.shape) == 1, 'probs must be one-dimensional'
            next_word_probs = next_word_probs ** t / np.sum(next_word_probs ** t) # опционально

            if sample:
                next_word = np.random.choice(vocab, p=next_word_probs) 
            else:
                next_word = vocab[np.argmax(next_word_probs)]
            caption_prefix[0] = caption_prefix[0]+ ' '+ next_word

            # RNN-ка сгенерила символ конца предложения, расходимся
            #if next_word == <ваш символ конца предложения>:
            if next_word == EOS:
                break
            
    return caption_prefix





