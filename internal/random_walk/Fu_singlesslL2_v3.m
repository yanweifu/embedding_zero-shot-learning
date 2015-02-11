function [sLabels, Zmat, sL1Labels] = Fu_singlesslL2_v3( L,  data, opts)

%
% using student t-distribution.
%
%
% [sLabels, cLabels] = mvssl_v2(v1, v2, v3, L, k, alpha, no)
% label propagation on multiple graphs respectively, and on the combined
% graph.
%
% INPUT:
% no - the number of testing data points. 
% pn - the prototype number (initial labeling number). In our case, each
%      class only has one prototype.
% d  - the dimension of the feature vector in the common latent space.
% data - d x (no+pn) data matrix of view-1
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
opts = getPrmDflt(opts,{'c',exp(0), 'k',30, 'alpha',0.3,'no',6180,'metric','Gaussian'},-1);

k = opts.k; alpha = opts.alpha; no = opts.no; c = opts.c;


 gNum = length(data);
% vdata = cell(gNum,1);
% vdata{1} = v1;
% vdata{2} = v2;
% vdata{3} = v3;


gK = cell(gNum, 1);
gvol = zeros(gNum, 1);
X_l2norm = cell(gNum, 1);
dN = length(L);
    
% do normalization--> to spherical space:
X = data;
XtX = X'*X;
data = sqrt(diag(XtX));    
%    gK{i} = X./(X_l2norm{i}*X_l2norm{i}');
data = X./repmat(data', size(X,1),1);

%% construct the multiple graphs.
%for i = 1:gNum,
    % compute the similarity, the inverse of the distance in our current
    % implementation.
dist = slmetric_pw(data, data,'sqdist');
  %  gK{i} = 1./dist; %1./(1+dist);
md = median(dist(:));

  
%      gK{i} = 1./(1+dist/(2*opts.cita(i)));
 %     gK{i} = 1./(1+dist/(c*md));
%gK = exp(-dist./(c*md));
gK = 1./(1+dist/(md));

% knn graph
%for i = 1:gNum,
K0 = gK;
Kn = zeros(dN, dN);
    for j = 1:dN,
        % collect the k-nearest neighbors
        [~, indx] = sort(K0(j,:), 'descend');
        ind = indx(2:k+1);
        % only store the k-nearest neighbors in the similarity matrix
        Kn(j, ind) = K0(j, ind);
    end;
    % compute the final symmetric similarity matrix
gKnn = (Kn+Kn')/2; clear Kn;

% compute the graph volumn
gvol = sum(gKnn(:));

%% L2-SSL
% graph normalization
D = 1./sqrt(sum(gKnn,2));
D = sparse(diag(D));
% compute the normalized graph Laplacian I-D^(-0.5)*W*D^(-0.5)
LAP = D*sparse(gKnn)*D;

Zmat = inv(eye(dN)-alpha*LAP);
% label propagation
sLabels = (1-alpha)*Zmat*L;
% compute the final class labels (transform from 1-0 representation to
% class id)
[~, sLabels] = max(sLabels(1:no,:), [], 2);
    

%% label propagation on each graph
l1para.eigs_refined = 0;
l1para.eigs_eps = 0.1;
l1para.lambda = 0.01; 
l1para.num_eigs = 100; % for l1-ssl
l1para.tolerance = 1e-6; 
l1para.maxIteration = 500;
l1para.STOPPING_GROUND_TRUTH = -1;
l1para.STOPPING_DUALITY_GAP = 1;
l1para.STOPPING_SPARSE_SUPPORT = 2;
l1para.STOPPING_OBJECTIVE_VALUE = 3;
l1para.STOPPING_SUBGRADIENT = 4;
l1para.stoppingCriterion = l1para.STOPPING_OBJECTIVE_VALUE;

    %% L1-SSL
    sL1Labels = l1ssl(LAP, L, no, l1para);


%% label propagation on multi-graph
%% L2-SSL
% D = 1./sqrt(sum(cW,2));
% D = sparse(diag(D));
% LAP = D*sparse(cW)*D;
% Zmat = inv(eye(dN)-alpha*LAP);
% cLabels = (1-alpha)*Zmat*L;
% [~, cLabels] = max(cLabels(1:no,:), [], 2);
% %% L1-SSL
% cL1Labels = l1ssl(LAP, L, no, l1para);