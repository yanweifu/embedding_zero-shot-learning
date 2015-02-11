function [sLabels, cLabels, Zmat] = Fu_mvsslL2_v3( L,  vdata, opts)

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
opts = getPrmDflt(opts,{ 'k',30, 'alpha',0.5,'no',6180},-1);
k = opts.k; alpha = opts.alpha; no = opts.no;


 gNum = length(vdata);
% vdata = cell(gNum,1);
% vdata{1} = v1;
% vdata{2} = v2;
% vdata{3} = v3;


gK = cell(gNum, 1);
gvol = zeros(gNum, 1);
X_l2norm = cell(gNum, 1);
dN = length(L);

% do normalization--> to spherical space:
for i = 1:gNum,
    X = vdata{i};
    XtX = X'*X;
    vdata{i} = sqrt(diag(XtX));    
%    gK{i} = X./(X_l2norm{i}*X_l2norm{i}');
    vdata{i} = X./repmat(vdata{i}', size(X,1),1);
end;
    
%% construct the multiple graphs.
for i = 1:gNum,
    X = vdata{i};
    % compute the similarity, the inverse of the distance in our current
    % implementation.
    dist = slmetric_pw(X, X,'sqdist');
    
md = median(dist(:));

%gK{i}= exp(-dist./(md));
gK{i} = 1./(1+dist./md);

    % compute the \|x_i\|_2^2
    XtX = X'*X;    
    X_l2norm{i} = diag(XtX);    
end;

% knn graph
for i = 1:gNum,
    K0 = gK{i};
    Kn = zeros(dN, dN);
    for j = 1:dN,
        % collect the k-nearest neighbors
        [~, indx] = sort(K0(j,:), 'descend');
        ind = indx(2:k+1);
        % only store the k-nearest neighbors in the similarity matrix
        Kn(j, ind) = K0(j, ind);
    end;
    % compute the final symmetric similarity matrix
    gK{i} = (Kn+Kn')/2; clear Kn;
end;

% compute the graph volumn
for i = 1:gNum,
    gvol(i) = sum(gK{i}(:));
end;

%% compute the graph's transition probabilities.
% we firstly compute the similarity between two graphs.
if gNum==3
temp_v12 = X_l2norm{1} + X_l2norm{2} - 2*diag(vdata{1}'*vdata{2});
sim_v12 = sum(1./sqrt(temp_v12)); %sum(1./(1+temp_v12(:)));

temp_v23 = X_l2norm{2} + X_l2norm{3} - 2*diag(vdata{2}'*vdata{3});
sim_v23 = sum(1./sqrt(temp_v23)); %sum(1./(1+temp_v23(:)));

temp_v13 = X_l2norm{1} + X_l2norm{3} - 2*diag(vdata{1}'*vdata{3});
sim_v13 = sum(1./sqrt(temp_v13)); %sum(1./(1+temp_v13(:)));
clear temp_v12 temp_v23 temp_v13;

% If we view the multi-graph as a single abstract graph with three
% vertexes, each vertex denotes each view, we can compute the degree of
% each vertex (each view).
v1_degree = sim_v12 + sim_v13;
v2_degree = sim_v12 + sim_v23;
v3_degree = sim_v13 + sim_v23;
vol_v123 = v1_degree + v2_degree + v3_degree;

% the graph priors (each degree devides the total volumn)
P_v1 = v1_degree/vol_v123;
P_v2 = v2_degree/vol_v123;
P_v3 = v3_degree/vol_v123;
PG = [P_v1, P_v2, P_v3];
% 0.3803    0.2682    0.3515

clear XtX X_l2norm;

elseif gNum==2
    temp_v12 = X_l2norm{1} + X_l2norm{2} - 2*diag(vdata{1}'*vdata{2});
    sim_v12 = sum(1./sqrt(temp_v12)); %sum(1./(1+temp_v12(:)));
    vol_v12= 2*sim_v12;
    P_v1 = gvol(1)/vol_v12;
    P_v2 = gvol(2)/vol_v12;
    
    residual = 1-P_v1-P_v2;
    
    PG =[P_v1+residual/2, P_v2+residual/2];
end

%% compute the multi-graph similarity matrix.  Zhou's mvssl has a very
%% simple equavelence as described in the previous illustation I sent to
%% you.
cW = zeros(dN, dN);
for i = 1:gNum,
    cW = cW + PG(i)*gK{i}/gvol(i);
end;

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

sLabels = cell(gNum,1);
sL1Labels = cell(gNum,1);
for i = 1:gNum,
    %% L2-SSL
    % graph normalization
    D = 1./sqrt(sum(gK{i},2));
    D = sparse(diag(D));
    % compute the normalized graph Laplacian I-D^(-0.5)*W*D^(-0.5)
    LAP = D*sparse(gK{i})*D;
    Zmat = inv(eye(dN)-alpha*LAP);
    % label propagation
    sLabels{i} = (1-alpha)*Zmat*L;
    % compute the final class labels (transform from 1-0 representation to
    % class id)
    [~, sLabels{i}] = max(sLabels{i}(1:no,:), [], 2);
    
    %% L1-SSL
%     sL1Labels{i} = l1ssl(LAP, L, no, l1para);
end;

%% label propagation on multi-graph
%% L2-SSL
D = 1./sqrt(sum(cW,2));
D = sparse(diag(D));
LAP = D*sparse(cW)*D;
Zmat = inv(eye(dN)-alpha*LAP);
cLabels = (1-alpha)*Zmat*L;
[~, cLabels] = max(cLabels(1:no,:), [], 2);
%% L1-SSL
% cL1Labels = l1ssl(LAP, L, no, l1para);