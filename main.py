from Process_Features import *
import joblib
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no2', type=float, default=False)
    parser.add_argument('--co', type=float, default=False)
    parser.add_argument('--so2', type=float, default=False)
    parser.add_argument('--pm25', type=float, default=False)
    parser.add_argument('--temp', type=float, default=False)
    parser.add_argument('--cloud', type=float, default=False)
    parser.add_argument('--amount', type=float, default=False) # 강수량 precipitation
    parser.add_argument('--press', type=float, default=False)
    parser.add_argument('--ws', type=float, default=False)
    parser.add_argument('--gust', type=float, default=False)
    parser.add_argument('--wd', type=str, default=False)
    parser.add_argument('--overall', type=str, default=False)
    parser.add_argument('--model', type=str, default='AiR_Predictor_RF.pkl')
    args = parser.parse_args()
    
    if not args.wd or not args.overall:
        raise ValueError("풍향과 전체적인 날씨는 꼭 입력해줘야 합니다.")
    
    test_sample = [args.no2, args.co, args.so2, args.pm25, args.temp, args.cloud, args.amount,
        args.press, args.ws, args.gust, args.wd, args.overall]

    # get before distribution
    dist = load_distribution('dist.txt')

    # fill drop attributes
    test_sample = fill_false(test_sample, dist)

    # preprocess & build for input features
    X = [process(test_sample, dist)]

    # load model
    model = joblib.load(args.model)
    print('<{}> model loaded'.format(args.model))
    
    # Classification
    y = model.predict(X).item()

    class_map = {0: '좋음', 1: '보통', 2: '나쁨', 3: '매우 나쁨'}
    
    print('>>>>>>>>>>')
    print('3시간 뒤 초미세먼지(PM2.5) 농도는 {} 단계로 예상됩니다'.format(class_map[y]))
    if y >= 2:
        print('초미세먼지용 마스크를 준비해서 외출하세요!')
    else:
        print('초미세먼지용 마스크까지는 필요하지 않습니다')
    
    return



if __name__ == '__main__':
    
    welcome = '''
    ====================
    AiR Predictor has been run.
    code by Junmay
    https://github.com/jonas-jun
    ====================
    '''
    print(welcome)

    intro = '''
    입력할 txt 데이터의 형태는 아래와 같습니다.
    no2, co, so2, 현재의 pm2.5 농도는 https://cleanair.seoul.go.kr/airquality/localAvg에서 확인할 수 있습니다.
    --no2: 현재의 이산화질소(NO2) 농도입니다.
    --co: 현재의 일산화탄소(CO) 농도입니다.
    --so2: 현재의 아황산가스(SO2) 농도입니다.
    --pm25: 현재의 초미세먼지(pm2.5) 농도입니다.
    --temp: 3시간 후의 예상 기온입니다.
    --cloud: 3시간 후의 예상 구름의 양입니다. (퍼센트 단위의 숫자)
    --amount: 3시간 후의 예상 강수량입니다.
    --press: 3시간 후의 기압입니다. 대부분 1000~1100의 숫자가 나옵니다.
    --ws: 3시간 후의 예상 풍속입니다 (m/s단위)
    --gust: 3시간 후의 예상 돌풍속도입니다. (m/s단위)
    --wd: 3시간 후의 예상 풍향(Wind Direction)입니다. 북서: 'NW', 서북서: 'WNW', 2개 단위일 때 남북+동서 형태, 한글 표현 그대로 대소문자 구분 없이 입력해주세요.
    --overall: 3시간 후의 전체적인 날씨입니다. 비나 눈이 오는 등 강수가 있으면 1, 그렇지 않으면 0 입니다.
    
    - 결측값이 있다면 2008-2018 기간의 평균치로 계산됩니다.
    - 다만 Categorical 데이터인 전체적인 날씨와 풍향은 꼭 입력해주세요. 평균치가 없습니다.
    '''
    print(intro)

    main()