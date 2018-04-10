# !/usr/bin/env
# -*- coding:utf-8 -*-
import math
import operator


# 计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # 遍历每个实例，统计target的频数
        currentLabel = featVec[-1]  # 每个实例最后一列是target
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


# 定义一个函数来划出选定特征之外的数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 计算条件熵
def calcConditionalEnt(dataSet, labels):
    conditionEnt = {}
    uniqueVals = []
    for data in dataSet:
        uniqueVals.append(data[:-1])  # 除最后一列之外的列为特征集
    n = len(uniqueVals[0])
    for value in uniqueVals:
        for i in xrange(0, n):
            conditionEnt[labels[i]] = 0
            subDataSet = splitDataSet(dataSet, i, value[i])
            prob = len(subDataSet) / float(len(dataSet))
            conditionEnt[labels[i]] += prob * calcShannonEnt(subDataSet)
    return conditionEnt


# 计算信息增益
def calcInformationGain(dataSet, labels):
    sEnt = calcShannonEnt(dataSet)
    cEnt = calcConditionalEnt(dataSet, labels)
    infoGain = {}
    for k, v in cEnt.iteritems():
        infoGain[k] = sEnt - v
    return infoGain


# 计算信息增益比
def calcInfGainRatio(dataSet, labels):
    sEnt = calcShannonEnt(dataSet)
    cEnt = calcConditionalEnt(dataSet, labels)
    infoGainRatio = {}
    for k, v in cEnt.iteritems():
        infoGainRatio[k] = (sEnt - v) / sEnt
    return infoGainRatio


# 抄一个网上的数据集
def createDataSet():
    dataSet = [['youth', 'no', 'no', 1, 'refuse'],
               ['youth', 'no', 'no', '2', 'refuse'],
               ['youth', 'yes', 'no', '2', 'agree'],
               ['youth', 'yes', 'yes', 1, 'agree'],
               ['youth', 'no', 'no', 1, 'refuse'],
               ['mid', 'no', 'no', 1, 'refuse'],
               ['mid', 'no', 'no', '2', 'refuse'],
               ['mid', 'yes', 'yes', '2', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '2', 'agree'],
               ['elder', 'yes', 'no', '2', 'agree'],
               ['elder', 'yes', 'no', '3', 'agree'],
               ['elder', 'no', 'no', 1, 'refuse'],
               ]
    labels = ['age', 'working?', 'house?', 'credit_situation']
    return dataSet, labels


# 选择信息增益最大的特征
def chooseByID3(dataSet, labels):
    infoGain = calcInformationGain(dataSet, labels)
    bestFeature = ''
    max = 0
    for k, v in infoGain.items():
        if (v > max):
            max = v
            bestFeature = k
    return bestFeature  # 返回最优特征对应的维度


# 选择信息增益比最大的特征
def chooseByC45(dataSet, labels):
    infoGainRatio = calcInfGainRatio(dataSet, labels)
    bestFeature = ''
    max = 0
    for k, v in infoGainRatio.items():
        if (v > max):
            max = v
            bestFeature = k
    return bestFeature  # 返回最优特征对应的维度


# 投票决定叶节点分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:  # 统计所有类标签的频数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 排序
    return sortedClassCount[0][0]


# 递归创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 第一个递归结束条件：所有的类标签完全相同
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 第二个递归结束条件：用完了所有特征
    # bestFeatLabel = chooseByCID3(dataSet, labels)  # 最优划分特征
    bestFeatLabel = chooseByC45(dataSet, labels)  # 最优划分特征
    bestFeat = labels.index(bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 使用字典类型储存树的信息
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 复制所有类标签，保证每次递归调用时不改变原始列表的内容
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    tree = createTree(dataSet, labels)
    import drawTree

    drawTree.createPlot(tree)
