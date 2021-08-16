#-*- coding: utf-8 -*-

def load_distribution(file):
    result = dict()
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key, mean, std = line.strip().split(',')
        result[key] = [float(mean)]
        result[key].append(float(std))
    return result

def fill_false(sample, dist):
    
    keys = list(dist.keys())
    for i in range(len(keys)):
        if not sample[i]:
            sample[i] = dist[keys[i]][0] # 평균치로 채우기
    return sample

# 바람 변환
def cat_wind(val):
    wind_map = {'NW': 'W', 'WSW': 'W', 'WNW': 'W', 'W': 'W', 'SW': 'W', 'SSW': 'S', 'E': 'E', 'ENE': 'E',
           'ESE': 'E', 'NNW': 'N', 'SE': 'E', 'S': 'S', 'SSE': 'S', 'NE': 'E', 'NNE': 'N', 'N': 'N'}
    return wind_map[val.upper()]

def encode_wind(val):
    wind_map_int = {'W': 0, 'E': 1, 'S': 2, 'N': 3}
    return wind_map_int[val]

# 날씨 변환
def cat_overall(val):
    rain = set(['drizzle', 'rain', 'snow', 'sleet', '비', '눈'])
    if set(val.lower().split()) & rain:
        return 1
    else:
        return 0

# 풍속 조정, gust와 wind_speed
def mps_to_mph(spd):
    return float(spd*3600*0.000621)

def outlier(val, maximum):
    return max(val, maximum)

def process(sample, dist):
    keys = list(dist.keys())
    standard = [0,1,2,7] # standardization idx: NO2, CO, SO2, pressure
    for i in standard:
        sample[i] = (sample[i] - dist[keys[i]][0]) / dist[keys[i]][1] # get Z
    # 날씨
    if type(sample[11]) == str:
        sample[11] = cat_overall(sample[11])
    # 풍향 변환
    sample[10] = encode_wind(cat_wind(sample[10]))    
    # 풍속 변환, 8,9 wind_speed, gust
    sample[8] = outlier(mps_to_mph(sample[8]), 20)
    sample[9] = outlier(mps_to_mph(sample[9]), 30)
    return sample

