# ds-with-mln
Implementation of Distant Supervision with Markov Logic Network for Korean Language<br />
Reference : [Han, Xianpei, and Le Sun. "Global distant supervision for relation extraction." AAAI. 2016.](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12006)

## Prerequisites
- Python 3.5+
	- numpy 1.13.0+
	- scikit-learn 0.18.2+
- [Alchemy 1.0](https://alchemy.cs.washington.edu/)

## 사용법
### 1. 설정
코드를 clone 받으신 후 config_sample.py 를 config.py로 복사합니다.
config.py 파일 안에 data_path는 이 프로젝트의 data 디렉토리 위치,
config.py 파일 안에 alchemy_path는 Alchemy 1.0의 binary 파일 디렉토리 위치를 설정해줍니다.

`config.py 예시`
```python
# data 폴더
data_path = './data/'

# alchemy 경로
alchemy_path = '/home/han0ah/alchemy/bin/'
```
### 2. train 및 test 파일
- train 파일 : ./data/train_data
- test 파일 : ./data/test_data

train 및 test 파일 포맷은 각 line이 데이터 하나를 의미하며 
'sbj[\t]obj[\t]relation[\t]문장' 의 형태입니다.
문장에서 subject entity와 object entity 위치는 각각
' << _sbj_ >> ' 와 ' << _obj_ >> ' placeholder로 적혀있습니다.

`train 및 test 파일 데이터 예시`
```python
아이유	가수	occupation	연예인  << _sbj_ >> 는 대한민국의  << _obj_ >> 이다.
```

### 3. Trainig
train.py 파이썬 스크립트를 실행시키면 됩니다.
```
python3 train.py
```

### 4. Test
1) test.py 파이썬 스크립트를 실행시킵니다.

```
python3 test.py
```
2) ./data/prec_recall_per_prop.txt 에 성능측정 결과파일이 생성됩니다.
각 라인은 'Relation[\t]Precision[\t]Recall[\t]F1-Score[\t]전체 개수' 형태로
Relation별 성능이 측정되어 있고 마지막 줄에 average 성능이 적혀있습니다.

`./data/prec_recall_per_prop.txt 예시`
```python
occupation	1.0	1.0	1.0	2
channel	0.5	0.5	0.5	2
average	0.75	0.75	0.75
```

3) ./data/prediction_result.txt 에 데이터별 Prediction 파일이 출력됩니다.
각 라인은 데이터별로
'sbj[\t]obj[\t]gold relation[\t]predicted relation[\t]confidence[\t]문장' 형태로
되어있습니다.

`./data/prediction_result.txt 예시`
```python
정기고	가수	occupation	occupation	0.9810	큐빅()은 대한민국 힙합 뮤지션이자  << _obj_ >> 인  << _sbj_ >> 의 예명이다.
```

### 5. 문서에서 관계추출

1) 정유성 연구원의 전처리 모듈(L2K-pack) output 중 
MLN용 파일을(wiki_ex_PL7.txt) 본 프로젝트의 './data/input' 으로 복사합니다.

2) run.py 파이썬 스크립트를 실행시키면 본 프로젝트 './data/output' 파일에 결과가 생성됩니다.

```
python3 run.py
```

3) './data/output' 파일의 포맷은 다른 프로젝트와 일치합니다 아래와 같습니다.
각 라인이 sbj[\t]예측한Relation[\t]obj[\t].[\t]confidence[\t]문장 
으로 이루어져있습니다.
`./data/output 예시`
```python
애플_(기업)	foundedBy	스티브_워즈니악	.	0.992171806968	애플_(기업) 은 스티브_잡스 와 스티브_워즈니악 과 로널드_웨인 이 1976년에 설립한 컴퓨터 회사 이다.
```

## 주의사항
- Markov Logic Network 는 속도 문제가 있습니다. 
- 따라서 다음 정도의 데이터 사이즈가 하루 안에 실험 가능한 범위 입니다.
	- Training 12,000개 데이터 내외, 30개 Relation (Weight Learning 시간 약 19시간)
	- Test 5,000개 데이터 내외, 30개 Relation (Inference 시간 약 5시간)

