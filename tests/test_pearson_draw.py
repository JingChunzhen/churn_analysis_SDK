import pandas as pd
from pyecharts import Line

'''
对excel表中的某列进行排序
对排序之后的结果分割
对分割之后的结果绘制pearson曲线，并计算pearson系数
'''

def draw(file_in, file_out):
    '''
    '''
    df = pd.read_csv(file_in, encoding='utf-8')
    print(type(df['留存比']))
    print(df['留存比'].corr(df['随后中位数时间间隔']))
    
    # 对每一条数据存如列表中
    l1 = df['留存比'].tolist()
    l1 = [e * 1000 for e in l1]
    l2 = df['随后中位数时间间隔'].tolist()

    attr = [i for i in range(len(l1))]

    line = Line("")
    line.add("流失留存比 * 1000", attr, l1)
    line.add("卡顿时间", attr, l2)

    line.show_config()
    line.render(file_out)

if __name__ == "__main__":
    '''
    0.309806861451
    0.465331387018
    0.0989582758376
    '''
    draw('../output/0-600.csv', '../output/0-600.html')
    draw('../output/600-1200.csv', '../output/600-1200.html')
    draw('../output/1200-2400.csv', '../output/1200-2400.html')
    pass