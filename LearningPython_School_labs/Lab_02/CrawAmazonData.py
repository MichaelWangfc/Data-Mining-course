import requests

def getHTMLText(url):
    try:
        kv = {'user-agent':'Chrome/63'}
        r = requests.get(url,timeout=30,headers = kv)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text[800:1000]
    except:
        print('爬取错误')


url = 'https://www.amazon.cn/dp/B071WMZ43K/'
print(getHTMLText(url))



