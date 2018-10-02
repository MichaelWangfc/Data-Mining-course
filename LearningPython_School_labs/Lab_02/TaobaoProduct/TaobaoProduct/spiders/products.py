# -*- coding: utf-8 -*-
import scrapy
import re

class ProductsSpider(scrapy.Spider):
    name = 'products'
    
    #allowed_domains = ['taobao.com']
    
    #start_urls = [] #初始URL地址

    #生成URL地址
    def start_requests(self):
        url_head = 'https://s.taobao.com/search?q=手机&s='
        for i in range(10):
            url = url_head + str(i*48)
            yield scrapy.Request(url,callback =self.parse )

    
    #配置获取页面后的解析方式
    def parse(self, response):
        infoDict ={} #初始化生成返回的item的空字典

        html = response.text
        
        #解析网页内容，得到有效信息
        item_ls = re.findall(r'"title":".*?"',html)
        price_ls = re.findall(r'"price":"[\d.]*"',html)

        #生成字典型的item 
        for i in range(len(item_ls)):
            item = eval(item_ls[i].split(':')[1])
            price = eval(price_ls[i].split(':')[1])
            infoDict[item]=price

        yield infoDict

