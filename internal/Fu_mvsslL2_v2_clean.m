function [sLabels, cLabels,   Zmat, cW,PG] = Fu_mvsslL2_v2_clean( L,  vdata, opts)

% [sLabels, cLabels] = mvssl_v2(v1, v2, v3, L, k, alpha, no)
% label propagation on multiple graphs respectively, and on the combined
% graph.
%
% INPUT:
% no - the number of testing data points. 
% pn - the prototype number (initial labeling number). In our case, each
%      class only has one prototype.
% d  - the dimension of the feature vector in the common latent space.
% v1 - d x (no+pn) data matrix of view-1
% v2 - d x (no+pn) data matrix of view-2
% v3 - d x (no+pn) data matrix of view-3
%      We put the feature vectors of the prototype instances to 
%      the end of the data matrix on each view.
% L  - (no+pn) x pn labeling matrix. We use the 1-0 representation for the initial labels.
% k  - the parameter for k-nearest-neighbor graph construction.
% alpha - the parameter of the manifold regularization term in label propagation.
% 
% OUTPUT:
% sLabels - label propagation result on each view (model)
% cLabels - label propagation result on the multi-graph

% setting default parameters:
% cita is the normalization parameters for each graph.
opts = getPrmDflt(opts,{ 'k',30, 'alpha',0.5,'no',6180, 'similarity_measure','Gaussian'},-1);
k = opts.k; alpha = opts.alpha; no = opts.no;

gNum = length(vdata);

gK = cell(gNum, 1);
gvol = zeros(gNum, 1);
X_l2norm = cell(gNum, 1);
dN = length(L);

% do normalization--> to spherical space:
for i = 1:gNum,
    tmp = l2norm(vdata{i}');
    vdata{i} =tmp';
end;
clear tmp;
    
%% construct the multiple graphs.
for i = 1:gNum,
    X = vdata{i};
    % compute the similarity, the inverse of the distance in our current
    % implementation.
    dist = slmetric_pw(X, X,'sqdist');
    
    if strcmp(opts.similarity_measure,'Gaussian')
        md = median(dist(:));
        gK{i}= exp(-dist./(md));
    elseif strcmp(opts.similarity_measure,'student-t')
        % student-t distribution
        gK{i} =1./(1+dist.^2/(2*md));
    end
end

% knn graph
for i =1:gNum
    gKnn{i} = filterKnn(gK{i},opts);
end

% compute the graph volumn
for i = 1:gNum,
    gvol(i) = sum(gKnn{i}(:));
end;

%% compute the graph's transition probabilities.
% we firstly compute the similarity between two graphs.
if gNum==3

PG = combineWeight(vdata);

elseif gNum==2
    temp_v12 = X_l2norm{1} + X_l2norm{2} - 2*diag(vdata{1}'*vdata{2});
    sim_v12 = sum(1./sqrt(temp_v12)); %sum(1./(1+temp_v12(:)));
    vol_v12= 2*sim_v12;
    P_v1 = gvol(1)/vol_v12;
    P_v2 = gvol(2)/vol_v12;
    
    residual = 1-P_v1-P_v2;
    
    PG =[P_v1+residual/2, P_v2+residual/2];
end

cW = zeros(dN, dN);
for i = 1:gNum,
    cW = cW + PG(i)*gKnn{i}/gvol(i);
end;

%% label propagation on each graph
sLabels = cell(gNum,1);

%% label propagation on multi-graph
D = 1./sqrt(sum(cW,2));
D = sparse(diag(D));
LAP = D*sparse(cW)*D;
Zmat = inv(eye(dN)-alpha*LAP);
cLabels = (1-alpha)*Zmat*L;
[~, cLabels] = max(cLabels(1:no,:), [], 2);
%% L1-SSL
% cL1Labels = l1ssl(LAP, L, no, l1para);