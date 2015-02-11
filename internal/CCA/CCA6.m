function [Wx, D,XX, index] = CCA6(D, Vd, Ad, O, Vo, Ao)

% implements the 4 view CCA that takes 4 views as input
% X is visual feature data
% T is tag SVD results
% T is the clustering indexes feature
% 
% XX, all combined features;
% index, for all features
% extended by Yanwei Fu


XX = [D, Vd, Ad, O, Vo, Ao];
index = [ones(size(D,2),1);ones(size(Vd,2),1)*2;ones(size(Ad,2),1)*3; ...
            ones(size(O,2),1)*4;ones(size(Vo,2),1)*5;ones(size(Ao,2),1)*6;];

[V, D] = MultiviewCCA(XX, index, 0.0001);
Wx = V;































