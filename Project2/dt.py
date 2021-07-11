import sys
import numpy as np
import os
import itertools
import math


class Node:

    def __init__(self, attribute, data):
        self.attribute = attribute
        self.data = data
        self.child = []


class DecisionTree:

    def __init__(self, trainFile, testFile, resultFile):
        self.trainfile = open(trainFile, 'r')
        self.testfile = open(testFile, 'r')
        self.resultfile = open(resultFile, 'w')
        self.attribute = []
        self.trainSet = []
        self.Class = dict()
        self.root = None
        self.testSet = []


    # Parse Input file and get Training Dataset
    def parseLine(self):
        lines = self.trainfile.readlines()

        file_length = len(lines)
        
        for line in range(file_length):

            if line == 0:
                temp_attribute = ''
                for parse in lines[line]:
                    if parse != '\t' and parse != '\n':
                        temp_attribute += parse

                    else:
                        self.attribute.append(temp_attribute)

                        temp_attribute = ''

                print(self.attribute)

            
            else:
                temp_str = ''
                parsed = []

                for parse in lines[line]:
                    if parse != '\t' and parse != '\n':
                        temp_str += parse

                    else:
                        parsed.append(temp_str)

                        temp_str =''

                # Last line of the file doesn't contain '\t' or '\n'
                if line == lines[-1]:
                    parsed.append(temp_str)

                #print("parsed line : " , parsed)

                self.trainSet.append(parsed)
                
                #print(parsed)

                if parsed[-1] not in self.Class:
                    self.Class[parsed[-1]] = 1

                else:
                    self.Class[parsed[-1]] += 1

                #print(self.Class)

        #print(self.trainSet)
        print(self.Class)

        self.root = Node(None, self.trainSet)

        attr_arr = self.attribute[:-1]

        self.classify(self.root, attr_arr)

    # Get Entropy from the Node
    def getEntropy(self, labels):

        label_num = sum(labels.values())

        entropy = 0

        for p in labels.values():
            
            entropy += - (p / label_num) * math.log2(p / label_num)

        return entropy


    # Calculate Entropy and Information Gain and Classify the Node
    # node : Node Class Object
    def classify(self, node, attributes):
        #print("Candidate Attributes : ", attributes)
        attr_len = len(attributes)
        tup_len = len(node.data)
        entropy_before = 0
        entropy_list = []

        labels = dict()

        max_info_gain = 0

        chosen_attr = ""

        # Quit Classify Function if no attirubtes left
        if len(attributes) == 0:
            return

        # Calculate Entropy of the Node before classification
        for tup in node.data:

            if tup[-1] not in labels:
                labels[tup[-1]] = 1

            else:
                labels[tup[-1]] += 1

        entropy_before = self.getEntropy(labels)

        # Quit Classify Function if entropy equals 0 (Classified Homogeneously)
        if entropy_before == 0:
            return

        # Try Classifying by attributes and calculate Information Gain
        for attr in attributes:

            entropy_after = 0
            info_gain = 0

            temp_class_list = []

            attr_idx = self.attribute.index(attr)
            temp_class = dict()

            for tup in node.data:

                if tup[attr_idx] not in temp_class:

                    attr_dict = dict()

                    attr_dict[tup[-1]] = 1

                    temp_class[tup[attr_idx]] = attr_dict

                    temp_class_list.append(attr_dict)

                else:
                
                    if tup[-1] not in temp_class[tup[attr_idx]]:

                        temp_class[tup[attr_idx]][tup[-1]] = 1

                    else:

                        temp_class[tup[attr_idx]][tup[-1]] += 1


            for entropy in temp_class_list:

                temp_info = self.getEntropy(entropy)

                entropy_after += (sum(entropy.values()) / tup_len) * temp_info

            info_gain = entropy_before - entropy_after

            #print(attr, ": ", info_gain)

            if max_info_gain < info_gain:

                max_info_gain = info_gain

                chosen_attr = attr

        attr = chosen_attr
        #print(attr)
        attr_idx = self.attribute.index(attr)

        classifying = dict()

        # Start Classifying by attribute which has Max Info Gain
        for tup in node.data:

            if tup[attr_idx] not in classifying:

                classifying[tup[attr_idx]] = 1

                tup_list = []
                tup_list.append(tup)

                child_attribute = dict()
                child_attribute[attr] = tup[attr_idx]

                child_node = Node(child_attribute, tup_list)

                node.child.append(child_node)

            else:

                child_idx = list(classifying.keys()).index(tup[attr_idx])

                node.child[child_idx].data.append(tup)

        #print("Classifying Attribute :", attr)

        '''
        for child in node.child:

            print(child.attribute)
            print(child.data)
        '''

        cand_attribute = attributes[:]
        cand_attribute.remove(attr)
        #print(attributes)

        # Call Recursive classify function for child nodes
        for child in node.child:
            
            self.classify(child, cand_attribute)


    def parseTestFile(self):
        
        lines = self.testfile.readlines()

        file_length = len(lines)
        
        for line in range(file_length):

            if line > 0: 

                temp_str = ''
                parsed = []

                for parse in lines[line]:
                    if parse != '\t' and parse != '\n':
                        temp_str += parse

                    else:
                        parsed.append(temp_str)

                        temp_str =''

                # Last line of the file doesn't contain '\t' or '\n'
                if line == lines[-1]:
                    parsed.append(temp_str)

                #print("parsed line : " , parsed)

                self.testSet.append(parsed)
                
                #print(parsed)

        #print("Test Set : ", self.testSet)


    # Classify Test Data Set and Predict Labels
    def test(self):

        #print("Test Files")

        root = self.root

        for test_tup in self.testSet:

            #print("Test Tuple : ", test_tup)

            while True:

                #print("Unlimited loop?")

                find_node = False
                # Go to node which has correct attribute
                for child in root.child:

                    attr = list(child.attribute.keys())

                    attr = attr[0]

                    attr_idx = self.attribute.index(attr)

                    test_attr = test_tup[attr_idx]

                    if child.attribute[attr] == test_attr:

                        root = child
                        #print("Selected child node : ", root.attribute)
                        find_node = True
                        break

                # If current node is leaf node, predict class
                # If there is no appropriate node to go, majority voting
                if find_node == False or len(root.child) == 0:

                    #print("This is leaf node : ", root.attribute)
                    
                    label = dict()

                    for data in root.data:

                        if data[-1] not in label:

                            label[data[-1]] = 1

                        else:

                            label[data[-1]] += 1

                    max_val = max(list(label.values()))

                    for key in label.keys():

                        if label[key] == max_val:

                            test_tup.append(key)
                            #print(test_tup)

                            root = self.root

                            #print("")

                            break

                    break
               

    # Make Result File
    def printResultFile(self):

        for idx, attr in enumerate(self.attribute):
            attr = str(attr)

            if idx == len(self.attribute) - 1:
                
                attr += '\n'
                self.resultfile.write(attr)

            else:
                attr += '\t'
                self.resultfile.write(attr)
        
        for res in self.testSet:

            for idx, attr in enumerate(res):
                attr = str(attr)

                if idx == len(res) - 1:
                    attr += '\n'
                    self.resultfile.write(attr)

                else:
                    attr += '\t'
                    self.resultfile.write(attr)


    def closeFile(self):
        self.testfile.close()
        self.trainfile.close()



def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    result_file = sys.argv[3]

    print(train_file, test_file, result_file)

    dt = DecisionTree(train_file, test_file, result_file)
    dt.parseLine()
    dt.parseTestFile()
    dt.test()
    dt.printResultFile()
    dt.closeFile()

    
if __name__ == "__main__":
    main()