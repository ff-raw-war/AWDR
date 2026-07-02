[eigVec, temp, ev]=eig1(L, c, 0);
% 最优解P是由Z定义的L的c个特征向量构成的，对应于c个最小特征值。

labelPredict= kmeans(eigVec, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');

Clu_result = ClusteringMeasure1(label, labelPredict); % Tpami2022
