import numpy as np

class KMeans:
    def __init__(self):
        return self

    def RandomInitilization(self, X, k):
        self.Centers = X[np.random.choice(len(X), k)]

    def KmeansPlusPlusInitilization(self, X, k):
        self.Centers = [X[np.random.randint(0, len(X) - 1)]]

        for i in range(1, k):
            dist = np.array([np.linalg.norm(self.Centers - CurElem, axis=1).min() for CurElem in X])
            self.Centers.append(X[np.argmax(dist, axis = 0)])

    def AssignClusters(self, X):
        return np.array([np.argmin(np.linalg.norm(self.Centers - CurElem, axis = 1)) for CurElem in X])
                
    def MoveCenters(self, X, k, NewClusters):
        CurCenters = np.zeros_like(self.Centers)

        for i in range(k):
            CurCluster = X[NewClusters == i]
            CurCenters[i] = np.mean(CurCluster, axis = 0)

        return CurCenters

    def ElbowDist(self, X, k):
        # find mean sum of squared distance between clusters elements
        Sum = 0

        for i in range (k):
            CurCluster = X[self.Clusters == i]
            for CurElem in CurCluster:
                Sum += np.sum(np.linalg.norm(self.Centers[i] - CurElem, axis=0)**2)

        return Sum / k

    def TransformData(self, X, k, InitilizationType = 'random', NumberIterations = False, Find = False):
        # main fuction
        self.count = 0

        if InitilizationType == 'random':
            self.RandomInitilization(X, k)
        elif InitilizationType == 'kmeans_pp':
            self.KmeansPlusPlusInitilization(X, k)
        else:
            print('Initialization error')
        
        flag = False
        while not flag:
            NewClusters = self.AssignClusters(X)
            NewCenters = self.MoveCenters(X, k, NewClusters)
            self.count += 1
            if np.linalg.norm(NewCenters - self.Centers) >= 1e-6:
                self.Centers = NewCenters
                if self.count > 300:
                    print('Algorithm stuck')
                    break
            else:
                flag = True
                self.Clusters = NewClusters
                if Find:
                    self.AvSum = self.ElbowDist(X, k)
                if NumberIterations:
                    print(f'Num of iteration for which algorithm converged: {self.count}')

        return self