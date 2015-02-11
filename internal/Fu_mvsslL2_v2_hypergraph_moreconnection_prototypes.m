function [cLabels,   Zmat, cW] = Fu_mvsslL2_v2_hypergraph_moreconnection_prototypes( L,  vdata, opts)

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

sLabels=cell(1);

opts = getPrmDflt(opts,{ 'k',30,'hypk',30, 'alpha',0.5,'no',6180},-1);
k = opts.k; alpha = opts.alpha; no = opts.no;


gNum = length(vdata);

gK = cell(gNum, 1);
gvol = zeros(gNum, 1);
X_l2norm = cell(gNum, 1);
dN = length(L);

% do l2 normalization for each view --> to spherical space:
for i = 1:gNum,
    X = vdata{i};
    XtX = X'*X;
    vdata{i} = sqrt(diag(XtX));    
    vdata{i} = X./repmat(vdata{i}', size(X,1),1);
end
    
%% construct the multiple graphs.
for i = 1:gNum
    X = vdata{i};
    % compute the similarity, the inverse of the distance in our current implementation.
    dist = slmetric_pw(X, X,'sqdist');
    
    md = median(dist(:));
    gK{i}= exp(-dist./(md));

    % compute the \|x_i\|_2^2
    XtX = X'*X;    
    X_l2norm{i} = diag(XtX);    
end

% % knn graph
% for i = 1:gNum,
%     K0 = gK{i};
%     Kn = zeros(dN, dN);
%     for j = 1:no
%         % collect the k-nearest neighbors
%         [~, indx] = sort(K0(j,:), 'descend');
%         ind = indx(2:k+1);
%         % only store the k-nearest neighbors in the similarity matrix
%         Kn(j, ind) = K0(j, ind);
%     end;
%     
%     
%     for j = 1+no:dN
%         % collect the k-nearest neighbors
%         [~, indx] = sort(K0(j,:), 'descend');
%        ind = indx(2:opts.prototypeTime*k+1);
%     %    ind = indx(2:end);
%         % only store the 3*k-nearest neighbors in the similarity matrix
%         Kn(j, ind) = K0(j, ind);
%     end;
%     % compute the final symmetric similarity matrix
%     gK{i} = (Kn+Kn')/2; clear Kn;
% end


%% old method: compute the graph volume:
% %% compute the graph volume
 for i = 1:gNum,
     gvol(i) = sum(gK{i}(:));
 end
%%
% new method:  hypergraph to assign the piece-wise combination weights:
hyperk = opts.hypk;
for i=1:gNum
    X = vdata{i};
    for j = i+1:gNum
     Y = vdata{j};
     
       % first using view X to query view Y: 
       hypdistXY = slmetric_pw(X, Y,'sqdist');
       md = median(hypdistXY(:));
       hypgkXY{i,j}= exp(-hypdistXY./(md));
       
       % colect the k-nearest neighour
       for  z=1:dN %no
            [~, indx] = sort(hypgkXY{i,j}(z,:), 'descend');
            ind = indx(2:hyperk);
            heteKnXY{i,j}(z,ind) = hypgkXY{i,j}(z,ind);
       end
       
        % then using view Y to query view X: 
       hypdistYX = slmetric_pw(Y, X,'sqdist');
       md = median(hypdistYX(:));
       hypgkYX{i,j}= exp(-hypdistYX./(md));
       
       % colect the k-nearest neighour
       for  z=1:dN %no
            [~, indx] = sort(hypgkYX{i,j}(z,:), 'descend');
            ind = indx(2:hyperk);
            heteKnYX{i,j}(z,ind) = hypgkYX{i,j}(z,ind);
       end
    end
end

% similarity weights:
for i=1:gNum
    for j=i+1:gNum
        weightXY{i,j} = sum(heteKnXY{i,j},2)/hyperk;
        weightYX{i,j} = sum(heteKnYX{i,j},2)/hyperk;
    end
end

for i=1:gNum
    for j =i+1:gNum
     %weight{i,j} = sqrt(weightXY{i,j}.*weightYX{i,j});
        weight{i,j} = (weightXY{i,j}+weightYX{i,j})/2;

    end
end

v1_degree = weight{1,2}+weight{1,3};
v2_degree = weight{1,2}+weight{2,3};
v3_degree = weight{1,3} +weight{2,3};
vol_v123 = v1_degree + v2_degree + v3_degree;
P_v1 = v1_degree./vol_v123;
P_v2 = v2_degree./vol_v123;
P_v3 = v3_degree./vol_v123;

% generate piece-wise weight matrix:
% PGmatr{1} = P_v1*P_v1';
% PGmatr{2} = P_v2*P_v2';
% PGmatr{3} = P_v3*P_v3';

%  PGmatr{1} = (repmat(P_v1,1, size(P_v1,1)) +repmat(P_v1',size(P_v1,1),1))/2;
%  PGmatr{2} =  (repmat(P_v2,1, size(P_v2,1)) +repmat(P_v2',size(P_v2,1),1))/2;
%  PGmatr{3} = (repmat(P_v3,1, size(P_v3,1)) +repmat(P_v3',size(P_v3,1),1))/2;
PGmatr(:,:,1) = (repmat(P_v1,1, size(P_v1,1)) +repmat(P_v1',size(P_v1,1),1))/2;
PGmatr(:,:,2) =  (repmat(P_v2,1, size(P_v2,1)) +repmat(P_v2',size(P_v2,1),1))/2;
PGmatr(:,:,3) = (repmat(P_v3,1, size(P_v3,1)) +repmat(P_v3',size(P_v3,1),1))/2;

weightPG = normalise(PGmatr,3);


cW = zeros(dN, dN);
% for i = 1:gNum,
%     cW = cW + PGmatr(:,:,i).*gK{i}/gvol(i);
% end;

for i = 1:gNum,
    cW = cW + weightPG(:,:,i).*gK{i};
end;


% knn graph
%for i = 1:gNum,
    K0 = cW;
    Kn = zeros(dN, dN);
    for j = 1:no
        % collect the k-nearest neighbors
        [~, indx] = sort(K0(j,:), 'descend');
        ind = indx(2:k+1);
        % only store the k-nearest neighbors in the similarity matrix
        Kn(j, ind) = K0(j, ind);
    end;
    
    
    for j = 1+no:dN
        % collect the k-nearest neighbors
        [~, indx] = sort(K0(j,:), 'descend');
       ind = indx(2:opts.prototypeTime*k+1);
    %    ind = indx(2:end);
        % only store the 3*k-nearest neighbors in the similarity matrix
        Kn(j, ind) = K0(j, ind);
    end;
    % compute the final symmetric similarity matrix
    cW = (Kn+Kn')/2; clear Kn;
%end

%% compute the graph's transition probabilities.

% % debug: 
% cW = zeros(dN, dN);
% dPG=[0.4, 0.3, 0.2];
% for i = 1:gNum,
%     cW = cW + dPG(i)*gK{i}/gvol(i);
% end;

%% label propagation on each graph


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