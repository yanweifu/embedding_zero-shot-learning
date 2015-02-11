function [Wx, D,XX, index] = CCA9(X, Vx, Ax, D,Vd,Ad, O,Vo,Ao)

% implements the 4 view CCA that takes 4 views as input
% X is visual feature data
% T is tag SVD results
% T is the clustering indexes feature
% 
% XX, all combined features;
% index, for all features
% extended by Yanwei Fu


%T = full(T);
XX = [X, Vx, Ax, D,Vd,Ad, O,Vo,Ao];
% index = [ones(size(X,2),1);ones(size(T,2),1)*2;ones(size(Y,2),1)*3; ones(size(O,2),1)*4];
index = [ones(size(X,2),1);ones(size(Vx,2),1)*2;ones(size(Ax,2),1)*3;  ...
                ones(size(D,2),1)*4;ones(size(Vd,2),1)*5;ones(size(Ad,2),1)*6; ...
                ones(size(O,2),1)*7;ones(size(Vo,2),1)*8;ones(size(Ao,2),1)*9; ];

[V, D] = MultiviewCCA(XX, index, 0.0001);
Wx = V;


