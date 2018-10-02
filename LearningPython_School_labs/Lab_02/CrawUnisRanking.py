# coding: utf-8


import re
import requests
from bs4 import BeautifulSoup
import bs4

url = 'http://www.zuihaodaxue.cn/zuihaodaxuepaiming2018.html'
def getHTMLText(url):
    try:
        r = requests.get(url,timeout = 30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        print('爬取错误')
html = getHTMLText(url)



def fillUnivList(html):
    ulist = []
    soup = BeautifulSoup(html,'html.parser')
    for tr in soup.find('tbody').children:
        if isinstance(tr,bs4.element.Tag):
            tds = tr.find_all('td')
            ulist.append([tds[0].string,tds[1].string,tds[3].string])
    return ulist

ulist = fillUnivList(html)     




def printUnivList(ulist,num):
    tplt = '{0:3}\t{1:{3}^15}\t{2:^10}'    
    print(tplt.format('排名','学校名称','综合评分',chr(12288)))
    for i in range(num):
        u =ulist[i]
        print(tplt.format(u[0],u[1].strip(),u[2],chr(12288)))

       
printUnivList(ulist,10)






