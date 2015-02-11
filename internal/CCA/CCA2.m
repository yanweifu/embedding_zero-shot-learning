function [Wx, D] = CCA2(X, T)


% implements the 2 view CCA that takes 2 views as input

T = full(T);
XX = [X,T];
index = [ones(size(X,2),1);ones(size(T,2),1)*2];
[V, D] = MultiviewCCA(XX, index, 0.0001);
Wx = V;































