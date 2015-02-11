function [Wx, D,XX, index] = CCA3(X, T, Y)

% implements the 3 view CCA that takes 3 views as input
% X is visual feature data
% T is tag SVD results
% T is the clustering indexes feature
% 
% XX, all combined features;
% index, for all features
% written by Yunchao Gong


T = full(T);
XX = [X,T,Y];
index = [ones(size(X,2),1);ones(size(T,2),1)*2;ones(size(Y,2),1)*3];
[V, D] = MultiviewCCA(XX, index, 0.0001);
Wx = V;































