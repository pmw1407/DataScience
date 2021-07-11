import sys
import numpy as np
import os
import itertools

class Apriori:

    def __init__(self, readFile, writeFile):
        self.fread = open(readFile, 'r')
        self.fwrite = open(writeFile, 'w')
        self.trx = []
        self.max_item = 0
        self.freq_itemSet = []
        self.freq_pattern = dict()

    def printFile(self):
        lines = self.fread.readlines()

        file_length = len(lines)
        print("file length : ", file_length)

        for i in lines[0]:
            if i == '\n' or i == '\t':
                print("blank or endline")
            else:
                print(i)

    # Parse Input file and get Transactions
    def parseLine(self):
        lines = self.fread.readlines()

        file_length = len(lines)
        
        for line in lines:
            temp_str = ''
            parsed = []

            for parse in line:
                if parse != '\t' and parse != '\n':
                    temp_str += parse

                else:
                    parsed.append(int(temp_str))

                    if self.max_item < int(temp_str):
                        self.max_item = int(temp_str)
                    
                    temp_str =''

            # Last line of the file doesn't contain '\t' or '\n'
            if line == lines[-1]:
                    parsed.append(int(temp_str))

            #print("parsed line : " , parsed)

            self.trx.append(parsed)

        #print(self.trx)

        return self.trx

    def retMaxItem(self):
        return self.max_item

    def getRatio(self, val):
        return (val / len(self.trx)) * 100

    # Get Frequent Itemset
    def getItemSet(self, minSup):
        #print(minSup)

        # The number of elements of frequent item set
        freq_val = 1

        # First frequent set
        for itemList in self.trx:
            for item in itemList:
                item = (item,)
                if item not in self.freq_pattern.keys():
                    self.freq_pattern[item] = 1
                else:
                    self.freq_pattern[item] += 1

        first_set = []

        for key, value in self.freq_pattern.items():
            if self.getRatio(value) >= minSup:
                first_set.append(set(key))

        self.freq_itemSet.append(first_set)

        #print("First frequent ItemSet : ", self.freq_itemSet[0])

        # Second or more frequent set
        while True:
            freq_val += 1

            #print("frequent value : ", freq_val)

            freq_length = len(self.freq_itemSet[freq_val - 2])

            #print("freq_length : ", freq_length)

            comb_set = set()

            # Gathering all of the items of frequent itemset
            for i in range(freq_length):
                comb_set = comb_set | self.freq_itemSet[freq_val - 2][i]
            
            #print("Survived items : ", comb_set)
            #print("Survived item length : ", len(comb_set))
            
            # Self Join
            comb_set = list(itertools.combinations(comb_set, freq_val))

            #print(comb_set)
            #print("After self join : ", len(comb_set))

            # Change tuple to set
            for i in range(len(comb_set)):
                comb_set[i] = set(comb_set[i])

            #print(comb_set)

            # Pruning
            # Copy comb_set list to iterate for loop
            temp_comb_set = comb_set[:]

            for candidate in temp_comb_set:
                
                candset = list(itertools.combinations(candidate, freq_val - 1))

                #print("candset : ", candset)

                for i in range(len(candset)):

                    prune = 1

                    # Change tuple to set
                    candset[i] = set(candset[i])
                    
                    for prev in self.freq_itemSet[freq_val - 2]:

                        if prev == candset[i]:
                            prune = 0
                    
                    # Delete candidate from comb_set list
                    if prune == 1:
                        comb_set.remove(candidate)
                        break

            #print("After pruning : ", len(comb_set))

            # Making k + 1 frequent itemset
            for candidate_set in comb_set:

                #print(candidate_set)

                for itemList in self.trx:

                    itemList = set(itemList)

                    # Check if transaction contains the candidate of k + 1 frequent itemset
                    if candidate_set.issubset(itemList):

                        temp_candidate_set = list(candidate_set)
                        temp_candidate_set.sort()

                        if tuple(temp_candidate_set) not in self.freq_pattern.keys():
                            self.freq_pattern[tuple(temp_candidate_set)] = 1
                        else:
                            self.freq_pattern[tuple(temp_candidate_set)] += 1

            # k + 1 frequent itemset
            nth_set = []

            # Check if candidates satisfiy minimum support
            for key, value in self.freq_pattern.items():
                if len(key) == freq_val and self.getRatio(value) >= minSup:
                    nth_set.append(set(key))

            self.freq_itemSet.append(nth_set)

            #print(freq_val, "th frequent itemset : ", nth_set, "length : ", len(nth_set))

            if len(nth_set) == 0:
                break

        #print(self.freq_pattern)
        #print(len(self.freq_pattern))


    def appAssociateRule(self):

        #print("Association Rule")

        iterate = len(self.freq_itemSet)

        for i in range(1, iterate):
            
            itemset = self.freq_itemSet[i]
            
            for freqset in itemset:

                freqset = list(freqset)
                freqset.sort()

                support = 0

                if len(freqset) == 1:
                    support = self.freq_pattern[(freqset,)]
                
                else:
                    support = self.freq_pattern[tuple(freqset)]

                for j in range(i):

                    comb = list(itertools.combinations(freqset, j + 1))

                    for item in comb:

                        item = list(item)
                        item.sort()
                        
                        confidence = support / self.freq_pattern[tuple(item)] * 100
                        res_support = self.getRatio(support)

                        counterpart = set(freqset) - set(item)

                        temp_item = set(map(int, item))
                        temp_counterpart = set(map(int, counterpart))

                        line = str(temp_item) + '\t' + str(temp_counterpart) + '\t'
                        line += str(format(res_support, ".2f")) + '\t' + str(format(confidence, ".2f")) + '\n'

                        self.fwrite.write(line)

    def closeFile(self):
        self.fread.close()
        self.fwrite.close()


def main():
    minSupport = sys.argv[1]
    fileName = sys.argv[2]
    resultFile = sys.argv[3]

    fopen = Apriori(fileName, resultFile)
    trxList = fopen.parseLine()
    max_item = fopen.getItemSet(float(minSupport))
    fopen.appAssociateRule()

    fopen.closeFile()



if __name__ == "__main__":
    main()