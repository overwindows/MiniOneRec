# MIND News Recommendation Competitions

해당 소스는 [MIND: MIcrosoft News Dataset](https://msnews.github.io/)에서 세계 2등을 수상할 당시 사용한 모델링 및 관련 소스입니다.    
앙상블 과정에서 여러 조합이 이루어졌기에 똑같은 결과가 나오지 않을 수 있다는 점 참고 부탁드립니다.     
[결과](https://gsai.snu.ac.kr/%EA%B0%95%EC%9C%A0-%EA%B5%90%EC%88%98%ED%8C%80-microsoft-%EC%A3%BC%EC%B5%9C-%EB%89%B4%EC%8A%A4-%EC%B6%94%EC%B2%9C-%EC%84%B8%EA%B3%84%EB%8C%80%ED%9A%8C/)     
[최종 paper](https://msnews.github.io/assets/doc/2.pdf)      

- 최종 모델
  [NRMS](https://wuch15.github.io/paper/EMNLP2019-NRMS.pdf)+[NAML](https://arxiv.org/pdf/1907.05576.pdf)+BERT (NNB)

## 학습하기

- 필수 패키지 설치
```bash
pip3 install -r requirements.txt
```
- Data 다운로드
  [MIND: MIcrosoft News Dataset](https://msnews.github.io/)

## File Directory 📂

```shell
MIND
├── 1. modeling
│   ├── NAML  # 기본 NRMS
│   ├── NRMS_BERT_Fixed_all_feature # NRMS Fine tuning
│   └── Using_BERT
│       ├── NRMS_BERT                    # NRMS & BERT Fine tuning 1
│       ├── NRMS_BERT_Fixed              # NRMS & BERT Fine tuning 2
│       ├── NRMS_BERT_Fixed_all_feature  # NRMS & BERT Fine tuning 3
│       └── NRMS_BERT_Fixed_all_feature2 # NRMS & BERT Fine tuning 4
└── 2. Ensemble

```

# Authors
``````
Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)