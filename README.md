# AiR_Predictor_foruse

3시간 후 서울의 초미세먼지 단계를 예측하는 모델을 배포합니다
----
## contents
code by Junmay
- main.py: 모델을 실행합니다. 주요 파라미터들에 대한 설명은 아래를 참고해주세요.
- AiR_Predictor_RF.pkl: Random Forest에 sampling weight를 주어서 학습시킨 모델입니다. 자세한 내용은 아래 DS project repo.를 참고해주세요.
- Process_Features.py: 데이터 전처리를 위한 함수들이 포함되어 있습니다.
- dist.txt: 2008-2018 서울의 날씨와 대기질 데이터의 분포를 나타내는 txt 파일입니다. 표준정규화(standardization)와 결측값 보완을 위해 사용합니다.

----
## log
2021-08-15 first commit  
2021-08-16 add function to adjust outliers(wind_speed, gust)  

  
----
데이터사이언스 프로젝트로 진행했던 서울시 초미세먼지 농도 예측 모델을 배포합니다. [DS project repo](https://github.com/jonas-jun/DS_air_quality)   

no2, co, so2, 현재의 pm2.5 농도는 [서울시 대기질 안내페이지](https://cleanair.seoul.go.kr/airquality/localAvg)에서 확인할 수 있습니다.
- --no2: 현재의 이산화질소(NO2) 농도입니다.
- --co: 현재의 일산화탄소(CO) 농도입니다.
- --so2: 현재의 아황산가스(SO2) 농도입니다.
- --pm25: 현재의 초미세먼지(pm2.5) 농도입니다.
- --temp: 3시간 후의 예상 기온입니다.
- --cloud: 3시간 후의 예상 구름의 양입니다. (퍼센트 단위의 숫자)
- --amount: 3시간 후의 예상 강수량입니다.
- --press: 3시간 후의 기압입니다. 대부분 1000~1100의 숫자가 나옵니다.
- --ws: 3시간 후의 예상 풍속입니다 (m/s단위)
- --gust: 3시간 후의 예상 돌풍속도입니다. (m/s단위)
- --wd: 3시간 후의 예상 풍향(Wind Direction)입니다. 북서: 'NW', 서북서: 'WNW', 2개 단위일 때 남북+동서 형태, 한글 표현 그대로 대소문자 구분 없이 입력해주세요.
- --overall: 3시간 후의 전체적인 날씨입니다. 비나 눈이 오는 등 강수가 있으면 1, 그렇지 않으면 0 입니다.

- 결측값이 있다면 2008-2018 기간의 평균치로 계산됩니다.
- 다만 Categorical 데이터인 전체적인 날씨와 풍향은 꼭 입력해주세요. 평균치가 없습니다.

-----
## to do
- 3시간 전이 아닌 12시간, 24시간 전에 예측할 수 있는 모델을 학습시켜보는 편이 더 유용할 수도 있어서 차후 진행해볼 생각입니다.
- 돌풍(gust)이나 구름(cloud) 지표를 직접 구해서 입력하기가 쉽지 않은 것 같습니다. 학습 성능에는 도움을 주었다고 생각하지만 실제 사용할 때는 불편을 야기할 수 있어서 이를 제외하고 학습시킨 모델의 성능을 테스트해봐야겠습니다.
