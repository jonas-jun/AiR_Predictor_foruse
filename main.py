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
    https://github.com/jonas-jun/AiR_Predictor_foruse
    ====================
    '''
    print(welcome)
    main()