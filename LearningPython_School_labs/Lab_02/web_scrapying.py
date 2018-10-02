
import requests

def getHTMLText(url):
    try:
        r = requests.get(url,timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        print('爬取错误')




if __name__ == '__main__':
    url = 'http://www.baidu.com'
    url = 'https://python123.io/ws/demo.html'
    html= getHTMLText(url)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html,'html.parser')
    print(soup.pretty())
    





