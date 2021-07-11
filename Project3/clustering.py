import sys
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

class DBSCAN:

    # Initiate parameters
    def __init__(self, n, Eps, MinPts):
        self.n = n
        self.Eps = Eps
        self.MinPts = MinPts
        self.ClusterIdx = 0
        self.noiseCluster = []
        self.neighCount = []


    # Read input.txt and set parameters
    def readData(self, input):
        self.df = pd.read_table(input, sep='\t', header=None, names=['index', 'x', 'y'])
        self.size = len(self.df)
        self.visit = np.full((self.size), False)
        self.noise = np.full((self.size), False)
        self.realm = np.full((self.size), 0)
        self.calDist()
        self.inputNum = input[0:6]

        #print(self.df[0:3])
        #print(self.df[['x', 'y']].values)


    # Calculate distance from whole points
    def calDist(self):
        x = self.df[['x']].values
        y = self.df[['y']].values

        x1, x2 = np.meshgrid(x, x)
        y1, y2 = np.meshgrid(y, y)

        self.dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


    # Clustering
    def cluster(self):
        for i in range(self.size):

            if self.visit[i] == False:
                self.visit[i] = True
                self.neighbor = self.getNeighbor(i)

                if len(self.neighbor) > self.MinPts:
                    self.ClusterIdx += 1
                    print("Start Clustering", self.ClusterIdx, "th cluster with", len(self.neighbor), "neighbors")
                    self.clusterExpand(i)
                    self.neighCount.append(len(self.neighbor))

                else:
                    self.noise[i] = True

        self.pruneCluster()
        self.df['cluster'] = self.realm
        self.writeFile()


    # Get Neighbors from idx
    def getNeighbor(self, idx):
        ret = []
        
        for i, temp in enumerate(self.dist[idx]):
            if temp < self.Eps:
                ret.append(i)
        
        return ret


    # Cluster Expansion through neighbors
    def clusterExpand(self, idx):
        self.realm[idx] = self.ClusterIdx
        tempIdx = 0

        while True:
            neighborIdx = self.neighbor[tempIdx]

            #print("Dormammu I've come to bargain")

            if self.visit[neighborIdx] == False:
                self.visit[neighborIdx] = True
                neighbor = self.getNeighbor(neighborIdx)

                if len(neighbor) > self.MinPts:
                    for i in neighbor:
                        if i not in self.neighbor:
                            self.neighbor.append(i)

            if self.realm[neighborIdx] == 0:
                self.realm[neighborIdx] = self.ClusterIdx

            tempIdx += 1

            if len(self.neighbor) <= tempIdx:
                return


    # Prune clusters more than self.n
    def pruneCluster(self):
        if self.n < self.ClusterIdx:

            for i in range(self.ClusterIdx - self.n):
                
                temp = self.neighCount.index(min(self.neighCount))
                self.noiseCluster.append(temp + 1)
                self.neighCount[temp] = self.size


    # Write output file
    def writeFile(self):
        fileIdx = 0
        for i in range(1, self.ClusterIdx + 1):

            if i not in self.noiseCluster:

                fileName = self.inputNum + "_cluster_" + str(fileIdx) + ".txt"
                df = self.df[self.df['cluster'] == i]
                df = df['index'].values
                np.savetxt(fileName, df, fmt='%d', delimiter='\n')
                fileIdx += 1

    

def main():
    input_file = sys.argv[1]
    n = int(sys.argv[2])
    Eps = int(sys.argv[3])
    MinPts = int(sys.argv[4])

    dbscan = DBSCAN(n, Eps, MinPts)
    dbscan.readData(input_file)
    dbscan.cluster()
    

if __name__ == "__main__":
    main()