# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html



#自定义的类 TaobaoProductPipeline
class TaobaoProductPipeline(object):
	#open_spider在开始进行爬虫时调用
    def open_spider(self, spider):
        self.f = open('TaobaoProduct.txt', 'w')
	
	#close_spider在结束爬虫时调用
    def close_spider(self, spider):
        self.f.close()
        
	#process_item在爬虫“运行”时调用
    def process_item(self, item, spider):
        try:
            for (k,v) in item.items():
                d ={}
                d['手机名称']=k
                d['手机价格']=v
                line = str(d)+ '\n'
                self.f.write(line)
        except:
            pass
