import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Point():
    def __init__(self,x,y):
        self.x=x
        self.y=y
    
    def show(self):
        print(self.x,"-",self.y)

def LI(p1,p2,x):
    lamda = (p2.y-p1.y)/(p2.x-p1.x)
    return round(p1.y+lamda*(x-p1.x),4)

start = Point(0,0.22)
p1 = Point(180,0.0)
p2 = Point(360,0.31)
p3 = Point(540,0.70)
p4 = Point(720,0.80)
p5 = Point(1400,0.82)
end = Point(1600,0.76)
pitch_angles = []

for x in range(start.x,end.x):
    if x <= p1.x:
        y = LI(start,p1,x)
    elif x >= p1.x and x <= p2.x:
        y = LI(p1,p2,x)
    elif x >= p2.x and x <= p3.x:
        y = LI(p2,p3,x)
    elif x >= p3.x and x <= p4.x:
        y = LI(p3,p4,x)
    elif x >= p4.x and x <= p5.x:
        y = LI(p4,p5,x)
    elif x >= p5.x and x <= end.x:
        y = LI(p5,end,x)
    point = Point(x,y)
    if x%2==0:
        noise = (0.0005 - 0) * np.random.random_sample(1) + 0
    else:
        noise = -((0.0005 - 0) * np.random.random_sample(1) + 0)
    pitch_angles.append(point.y+noise[0])

df = pd.DataFrame({'P_ANG':pitch_angles})
df.plot()
plt.ylim(0,1)
plt.show()
df.to_csv("sagami_maru_2.csv",index=False)