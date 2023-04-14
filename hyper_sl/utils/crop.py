import numpy as np

def crop(data):
    start_x, start_y = 75,77
    res_x, res_y = 890, 580
    
    data = data[start_y: start_y + res_y, start_x:start_x+res_x]
    
    return data
