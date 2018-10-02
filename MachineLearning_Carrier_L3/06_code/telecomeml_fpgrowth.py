#!-*- coding:utf-8 -*-
#用于添加注释
#1、读取数据# 不能用pandas的原因是每一行的列数不相等
import csv
import itertools
import pyfpgrowth as fp
import matplotlib as plt
def initData(destId):
    destDatalist=[]
    with open("D:\workspace\TelecomML\data\Correlation.CSV","r") as csv_file:
        csv_file1 = csv.reader(csv_file)
        for i, rows in enumerate(csv_file1):
            if   0<i:
                rowValue=rows[0].split(" ")
                if int(rowValue[1])==destId:
                    if len(rowValue)>2:
                        rowValue.remove(rowValue[0])
                        rowValue.remove(rowValue[0])
                        destDatalist.append(rowValue)
                else:
                    continue
    return destDatalist


# #1、获取类别数
def getlenRuleKinds(sortRuleslist):
    kindsdict={}
    if isinstance(sortRuleslist,dict):
        # print sortRuleslist.items()
        for rule in sortRuleslist.items():
            rulekey=rule[0]
            rulekeyitem=len(rulekey)+1
            if kindsdict.has_key(rulekeyitem):
                kindscount = kindsdict.get(rulekeyitem)
                kindsdict[rulekeyitem] = int(kindscount) + 1
            else:
                kindsdict[rulekeyitem] = 1

    elif isinstance(sortRuleslist,list):
        # print sortRuleslist
        for rule in sortRuleslist:
            # print rule
            rulekey = rule[0]
            rulekeyitem = len(rulekey) + 1
            if kindsdict.has_key(rulekeyitem):
                kindscount = kindsdict.get(rulekeyitem)
                kindsdict[rulekeyitem] = int(kindscount) + 1
            else:
                kindsdict[rulekeyitem] = 1

    kindlist=sorted(kindsdict.items(),key=lambda  item:item[0],reverse=True)
    for item in kindlist:
        if item[0]>2:
            kindsdict[item[0]-1]=item[0]*kindsdict[item[0]]+kindsdict[item[0]-1]

    return kindsdict

#
def getlenRules(sortRuleslist,rulelen):
    ruledict={}
    # print sortRuleslist
    if isinstance(sortRuleslist,dict):
        # print sortRuleslist.items()
        for rule in sortRuleslist.items():
            rulekey=rule[0]
            rulevalue = rule[1]
            # print rulekey
            if (len(rulekey)+1>=rulelen):
                rulecell = rulekey + rulevalue[0]
                for antecedent in itertools.combinations(sorted(rulecell), rulelen):
                    if ruledict.has_key(antecedent):
                        ruleItem_conf_new = rulevalue[1]
                        ruleItem_conf_old = ruledict[antecedent]
                        if isinstance(ruleItem_conf_old,tuple):
                            if ruleItem_conf_old[0] == ruleItem_conf_new[0]:
                                ruleItem_conf_new[1] = max(float(ruleItem_conf_new[1]), float(ruleItem_conf_old[1]))
                                ruledict[antecedent] = tuple(ruleItem_conf_new[0], ruleItem_conf_new[1])
                            else:
                                ruleItem_conf_value = float(ruleItem_conf_new[1]) + float(ruleItem_conf_old[1])
                                ruledict[antecedent] = ruleItem_conf_value
                        else:
                            ruleItem_conf_value=max(float(ruleItem_conf_new), ruleItem_conf_old)
                            ruledict[antecedent] = ruleItem_conf_value
                    else:
                        ruleItem_conf_new = rulevalue[1]
                        ruledict[antecedent] = ruleItem_conf_new
            else:
                continue
            # if len(rule[0])==rulelen:
            #     rulelist.append(rule)
        if len(ruledict)==0:
            print '不存在包含%d项集频繁项'% rulelen
            return None
        else:
            return sorted(ruledict.items(), key=lambda x: x[1], reverse=True)
    elif isinstance(sortRuleslist,list):
        # print sortRuleslist
        for rule in sortRuleslist:
            rulekey = rule[0]
            rulevalue = rule[1]
            # print rulekey
            if (len(rulekey) + 1 >= rulelen):
                rulecell = rulekey + rulevalue[0]
                for antecedent in itertools.combinations(sorted(rulecell), rulelen):
                    if ruledict.has_key(antecedent):
                        ruleItem_conf_new = rulevalue[1]
                        ruleItem_conf_old = ruledict[antecedent]
                        if isinstance(ruleItem_conf_old, tuple):
                            if ruleItem_conf_old[0] == ruleItem_conf_new[0]:
                                ruleItem_conf_new[1] = max(float(ruleItem_conf_new[1]), float(ruleItem_conf_old[1]))
                                ruledict[antecedent] = tuple(ruleItem_conf_new[0], ruleItem_conf_new[1])
                            else:
                                ruleItem_conf_value = float(ruleItem_conf_new[1]) + float(ruleItem_conf_old[1])
                                ruledict[antecedent] = ruleItem_conf_value
                        else:
                            ruleItem_conf_value = max(float(ruleItem_conf_new), ruleItem_conf_old)
                            ruledict[antecedent] = ruleItem_conf_value
                    else:
                        ruleItem_conf_new = rulevalue[1]
                        ruledict[antecedent] = ruleItem_conf_new
            else:
                continue
            # if len(rule[0])==rulelen:
            #     rulelist.append(rule)
        if len(ruledict) == 0:
            print '不存在包含%d项集频繁项' % rulelen
            return None
        else:
            return sorted(ruledict.items(),key=lambda x:x[1],reverse=True)

def Id2CardName(cardIdlist):
    cardName_dict={}
    card_dict = {'90063345': '腾讯大王卡', '90046637': '腾讯视频小王卡',
                 '90046638': '腾讯呢音频小王卡', '90065147': '滴滴大王卡',
                 '90065148': '滴滴小王卡', '90151621': '滴滴大橙卡',
                 '90151624': '滴滴小橙卡', '90109916': '蚂蚁大宝卡',
                 '90109906': '蚂蚁小宝卡', '90127327': '百度大神卡',
                 '90157593': '百度女神卡', '90138402': '招行大招卡',
                 '90157638': '哔哩哔哩22卡', '90151622': '微博大V卡',
                 '90163763': '饿了么大饿卡', '90199605': '懂我卡',
                 '90129503': '京东小强卡', '90215356': '阿里YunOS-9元卡'}
    for cardIds in cardIdlist:
        cardIdConf=cardIds[1]
        cardNameTuple=''
        for cardId in cardIds[0]:
            # print card_dict[cardId]
            cardNameTuple=cardNameTuple+card_dict[cardId]+','
        cardName_dict[cardNameTuple[0:-1]]=cardIdConf
    return cardName_dict
#
# #
# # print len(getlenRuleKinds(sortRules)),'\n'
# # print getlenRuleKinds(sortRules)
if __name__ == '__main__':
    destIddict = {571:'杭州', 574:'宁波', 577:'温州', 573:'嘉兴',
                  572:'湖州', 575:'绍兴', 579:'金华', 570:'衢州',
                  580:'舟山', 576:'台州', 578:'丽水'}
    for destid in destIddict.items():
        print '%s地区的电话套餐列表'%destid[1]
        dest_data=initData(destid[0])
        result1=fp.find_frequent_patterns(dest_data,100)
        rules=fp.generate_association_rules(result1,0.35)
        sortRules=sorted(rules.items(),key = lambda x:x[1][1],reverse = True)
        # print sortRules
        print "包含%d类套餐方案" % len(getlenRuleKinds(sortRules))
        kindkeys = getlenRuleKinds(sortRules).keys()
        import json
        for kindkey in kindkeys:
            print "************第%d类套餐*****************"%(kindkey-1)
            print "包含%d个套餐组合的频繁套餐规则："%kindkey
            rulesval = getlenRules(sortRules, kindkey)
            cardNameDict=Id2CardName(rulesval)
            print "规则总数%d个,分别是:"% len(cardNameDict),json.dumps(cardNameDict, ensure_ascii=False, encoding='utf-8')
            print "***************************************"
        print '\n'











