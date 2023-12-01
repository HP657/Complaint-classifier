import streamlit as st
from konlpy.tag import Okt
import numpy as np
import pandas as pd
import joblib
from gensim.models import Word2Vec
from tkinter.tix import COLUMN
from pyparsing import empty

#꾸미기
from PIL import Image

# Load your images
image1 = Image.open("logo.gif")
image2 = Image.open("slogan_txt(2022).png")
image3 = Image.open("ic_taegugki.png")

# Display images side by side using columns
col1, col2, col3 = st.columns(3)
col1.image(image1, caption='', use_column_width=True)
col2.image(image2, caption='', use_column_width=True)
col3.image(image3, caption='', use_column_width=False)



def get_sent_embedding(model, embedding_size, tokenized_words):
    feature_vec = np.zeros((embedding_size,), dtype="float32")
    n_words = 0
    for word in tokenized_words:
        if word in model.wv.key_to_index:
            n_words += 1
            # 임베딩 벡터에 해당 단어의 벡터를 더함
            feature_vec = np.add(feature_vec, model.wv[word])
    # 단어 개수가 0보다 큰 경우 벡터를 단어 개수로 나눠줌 (평균 임베딩 벡터 계산)
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

# 문장벡터 데이터 셋 만들기
def get_dataset(sentences, model, num_features):
    dataset = list()

    # 각 문장을 벡터화해서 리스트에 저장
    for sent in sentences:
        dataset.append(get_sent_embedding(model, num_features, sent))

    # 리스트를 numpy 배열로 변환하여 반환
    sent_embedding_vectors = np.stack(dataset)
    
    return sent_embedding_vectors

st.title('금정구 민원 접수기')
train = st.text_area('민원 내용을 작성해주세요. ')
button_clicked = st.button('접수하기')
okt = Okt()


model_filename = 'model.joblib'
RF_model = joblib.load(model_filename)

model_filename1 = 'word_model.joblib'
wv_model = joblib.load(model_filename1)




if button_clicked:
    
    token_okt = [okt.morphs(train)]

    train_data_vecs = get_dataset(token_okt, wv_model, 1000)

    train_pred = RF_model.predict(train_data_vecs)
    st.write(f'{train_pred[0]} 에 접수 되었습니다. ')
    
    happy3 = ['행정지원국 세무과','기획감사실','금정도서관','금정문화회관','동 행정복지센터','행정지원국 총무과','행정지원국 재무과','행정지원국 민원여권과','행정지원국 토지정보과','문화복지국 문화관광과','문화복지국 평생교육과','문화복지국 사회복지과','문화복지국 가족정책과','문화복지국 생활보장과','경제환경국 일자리경제과','경제환경국 도시재생과','경제환경국 자원순환과','경제환경국 환경위생과','경제환경국 공원녹지과','안전도시국 안전관리과','안전도시국 도시관리과','안전도시국 교통행정과','안전도시국 건설과','안전도시국 건축과','보건소 보건행정과','보건소 건강증진과']
