%function [sLabels, cLabels,   Zmat, cW] = Fu_hypergraph_debug( L,  vdata, opts)


sLabels=cell(1);

opts = getPrmDflt(opts,{ 'k',30,'hypk',30, 'alpha',0.5,'no',6180},-1);
k = opts.k; alpha = opts.alpha; no = opts.no;

gNum = length(vdata);

gK = cell(gNum, 1);
gvol = zeros(gNum, 1);
dN = length(L);

% do l2 normalization for each view --> to spherical space:
for i = 1:gNum,
     eachrow= l2norm(vdata{i}');
     vdata{i} = eachrow';
end
    
% construct the multiple graphs.
for i = 1:gNum
    X = vdata{i};
    % compute the similarity, the inverse of the distance in our current implementation.
    dist = slmetric_pw(X, X,'sqdist');
    md = median(dist(:));
    gK{i}= exp(-dist./(md));    
    zsgK{i} = doMatrxZscore(gK{i});
end

% get their knn graph
opts.prototypeTime = 9;
for i=1:gNum
    gKnn{i} = filterKnn(gK{i},opts);
    zsgKnn{i} = filterKnn(zsgK{i},opts);    
end

% results of combinations (no-zscore);
PG= combineWeight(vdata);
cW = zeros(dN, dN);
 w=[1, 1, 1];
for i = 1:gNum,
    cW = cW + w(i)*PG(i)*gKnn{i};
end
accgKnn = db_cW(cW,test_img_label,zsl_label,dN);

% results of combinations (with double zscore);
cW = zeros(dN, dN);
for i = 1:gNum,
    cW = cW + PG(i)*zsgKnn{i};
end
acczsgKnn = db_cW(cW,test_img_label,zsl_label,dN);

for i =1:gNum
    acczsgKnn_eachview(i,:) = db_cW(zsgKnn{i},test_img_label,zsl_label,dN);
    accgKnn_eachview(i,:) = db_cW(gKnn{i},test_img_label,zsl_label,dN);
end

%%
% new method:  hypergraph to assign the piece-wise combination weights:
hyperk = opts.hypk;
for i=1:gNum
    X = vdata{i};
    for j = i+1:gNum
     Y = vdata{j};
     
       % first using view X to query view Y: 
       hdistXY = slmetric_pw(X, Y,'sqdist');
       md = median(hdistXY(:));
       gKXY{i,j}= exp(-hdistXY./(md));
       
       % do zscore normalization: first row, then column:
       zsgXY{i,j} = doMatrxZscore(gKXY{i,j});
       
       % colect the k-nearest neighour
       zsgKnnXY{i,j} = filterKnn(zsgXY{i,j},opts);
           gKnnXY{i,j} = filterKnn(gKXY{i,j},opts);
       
        % then using view Y to query view X: 
       hdistYX = slmetric_pw(Y, X,'sqdist');
       md = median(hdistYX(:));
       gKYX{i,j}= exp(-hdistYX./(md));
       
       zsgYX{i,j} = doMatrxZscore(gKYX{i,j});
       
       % colect the k-nearest neighour
       zsgKnnYX{i,j} = filterKnn(zsgYX{i,j},opts);
       gKnnYX{i,j} = filterKnn(gKYX{i,j},opts);
       
       % results of each view:
      acczsgKnnYX{i,j}=db_cW(zsgKnnYX{i,j},test_img_label,zsl_label,dN);
      acczsgKnnXY{i,j}=db_cW(zsgKnnXY{i,j},test_img_label,zsl_label,dN);
      
      accgKnnXY{i,j} = db_cW(gKnnXY{i,j},test_img_label,zsl_label,dN);
      accgKnnYX{i,j} = db_cW(gKnnYX{i,j},test_img_label,zsl_label,dN);
    end
end
%% how to best combine them?
ocW =cW;

hypergph =[zsgKnnYX{1,2}; zsgKnnYX{1,3};zsgKnnYX{2,3}; ];
nodeidx=[1:6190];
simYX= Knn_hypergraph_sim(hypergph, nodeidx);
acczsgKnnYX = db_cW(simYX,test_img_label,zsl_label,dN);

wyx  =[1,1];
cwyx = ocW*wyx(1) + simYX*wyx(2);
accYX = db_cW(cwyx,test_img_label,zsl_label,dN);


hypergph =[zsgKnnXY{1,2}; zsgKnnXY{1,3};zsgKnnXY{2,3}; ];
nodeidx=[1:6190];
simXY= Knn_hypergraph_sim(hypergph, nodeidx);
acczsgKnnXY = db_cW(simXY,test_img_label,zsl_label,dN);

wxy =[1,1];
cwxy = ocW*wxy(1) + simXY*wxy(2);
accXY = db_cW(cwxy,test_img_label,zsl_label,dN);


hypergph =[zsgKnnYX{1,2}; zsgKnnYX{1,3};zsgKnnYX{2,3};zsgKnnXY{1,2}; zsgKnnXY{1,3};zsgKnnXY{2,3}; ];
nodeidx=[1:6190];
simYXXY= Knn_hypergraph_sim(hypergph, nodeidx);
acczsgKnnYXXY = db_cW(simYXXY,test_img_label,zsl_label,dN);

wyxxy =[1,1];
cwyxxy = ocW*wyxxy(1) + simYXXY*wyxxy(2);
accYXXY = db_cW(cwyxxy,test_img_label,zsl_label,dN);

%% brute-force search:
%wi=[1,1; 1.1,1;1.2,1;1.3,1;1.4,1; 1.5,1;1.6,1;1,1.1;1,1.2;1,1.3;1,1.4;1,1.5;1,1.6];
wi=[1,1.4;1,1.5;1,1.55;1, 1.57;1, 1.6; 1, 1.7; 1, 1.8; 1, 1.9; 1.55, 1; 1.6,1 ];

wi =CCV_normalize(wi,1);
for i=1:length(wi)
wyxxy =wi(i,:);
cwyxxy = cW*wyxxy(1) + simhyper*wyxxy(2);
accYXXY = db_cW(cwyxxy,test_img_label,zsl_label,dN);
mav(i) =max(accYXXY);
end


%%
[PG,temp_v12,temp_v23,temp_v13]= combineWeight(vdata);
cW = zeros(dN, dN);
for i = 1:gNum,
    cW = cW + PG(i)*gKnn{i};
end
[accgKnn, predicted ]= db_cW(cW,test_img_label,zsl_label,dN);

subplot(221); hist(temp_v12,[min(temp_v12):0.2:max(temp_v12)]);
subplot(222); hist(temp_v13,[min(temp_v13):0.2:max(temp_v13)]);
subplot(223); hist(temp_v23,[min(temp_v23):0.2:max(temp_v23)]);

num = size(vdata{1},2);    
% combine into a single graph:

%  the distance between the same nodes from graph 1 and graph 2.
temp_v12 = 2*ones(num,1)  - 2*sum(vdata{1}'.*vdata{2}',2);  
% we use t-student distribution:
thr =exp(-2);;
% t12= 1./sqrt(temp_v12); f12 = t12>thr;
t12= exp(-temp_v12); f12 = t12>thr;
sim_v12 = sum(t12.*f12);

temp_v23 = 2*ones(num,1) - 2*sum(vdata{2}'.*vdata{3}',2);
%t23 = 1./sqrt(temp_v23); f23 = t23>thr;
t23 = exp(-temp_v23); f23 = t23>thr;
sim_v23 = sum(t23.*f23); 

temp_v13 = 2*ones(num,1) - 2*sum(vdata{1}'.*vdata{3}',2);
% t13= 1./sqrt(temp_v13); f13 = t13>thr;
t13= exp(-temp_v13); f13 = t13>thr;
sim_v13 = sum(t13.*f13); 

v1_degree = sim_v12 + sim_v13;
v2_degree = sim_v12 + sim_v23;
v3_degree = sim_v13 + sim_v23;
vol_v123 = v1_degree + v2_degree + v3_degree;

% the graph priors (each degree devides the total volumn)
P_v1 = v1_degree/vol_v123;
P_v2 = v2_degree/vol_v123;
P_v3 = v3_degree/vol_v123;
PG = [P_v1, P_v2, P_v3];

%%
figure(2);
subplot(2,3,1)
imagesc(tmphypgkXY{1,2});
title('low->con (1,2)');
subplot(2,3,2)
imagesc(tmphypgkXY{1,3});
title('low->bin (1,3)');
subplot(2,3,3);
imagesc(tmphypgkXY{2,3});
title('con->bin (2,3)');

subplot(2,3,4)
imagesc(tmphypgkYX{1,2});
title('con->low(1,2)');
subplot(2,3,5)
imagesc(tmphypgkYX{1,3});
title('bin->low (1,3)');
subplot(2,3,6);
imagesc(tmphypgkYX{2,3});
title('bin->con (2,3)');


% figure(3);
% subplot(2,3,1)
% imagesc(zscore(hypgkXY{1,2}));
% title('low->con (1,2)');
% subplot(2,3,2)
% imagesc(zscore(hypgkXY{1,3}));
% title('low->bin (1,3)');
% subplot(2,3,3);
% imagesc(zscore(hypgkXY{2,3}));
% title('con->bin (2,3)');
% 
% subplot(2,3,4)
% imagesc(zscore(hypgkYX{1,2}));
% title('con->low(1,2)');
% subplot(2,3,5)
% imagesc(zscore(hypgkYX{1,3}));
% title('bin->low (1,3)');
% subplot(2,3,6);
% imagesc(zscore(hypgkYX{2,3}));
% title('bin->con (2,3)');


% figure(3);
% 
% subplot(2,2,1)
% col=zscore(hypgkXY{1,2});
% row=zscore(col');
% a= row';
% imagesc(a);
% title('low->con (1,2)');
% 
% subplot(2,2,2)
% row=zscore(hypgkXY{1,2}');
% col=zscore(row');
% b= col';
% imagesc(b);
% title('low->con (1,2)');
% 
% subplot(2,2,3)
% col=zscore(hypgkXY{1,2});
% 
% imagesc(col);
% title('zscore-col');
% 
% subplot(2,2,4)
% row=zscore(hypgkXY{1,2}');
% row= row';
% imagesc(row);
% title('zscore-col');

figure(3);

subplot(2,3,1)
row=zscore(hypgkXY{1,2});
col=zscore(row');
d=col';
imagesc(d);
title('low->con (1,2)');
subplot(2,3,2)
row = zscore(hypgkXY{1,3});
col=zscore(row');
d=col';
imagesc(d);

title('low->bin (1,3)');
subplot(2,3,3);
%imagesc();
row = zscore(hypgkXY{2,3});
col=zscore(row');
d=col';
imagesc(d);

title('con->bin (2,3)');

subplot(2,3,4)
row =zscore(hypgkYX{1,2});
col=zscore(row');
d=col';
imagesc(d);


title('con->low(1,2)');
subplot(2,3,5)
row =zscore(hypgkYX{1,3});
col=zscore(row');
d=col';
imagesc(d);


title('bin->low (1,3)');
subplot(2,3,6);
row =zscore(hypgkYX{2,3});

col=zscore(row');
d=col';
imagesc(d);

title('bin->con (2,3)');


figure(3);

subplot(2,3,1)
imagesc(heteKnXY{1,2});
title('low->con (1,2)');
subplot(2,3,2)
imagesc(heteKnXY{1,3});
title('low->bin (1,3)');
subplot(2,3,3);
imagesc(heteKnXY{2,3});
title('con->bin (2,3)');

subplot(2,3,4)
imagesc(heteKnYX{1,2});
title('con->low(1,2)');
subplot(2,3,5)
imagesc(heteKnYX{1,3});

title('bin->low (1,3)');
subplot(2,3,6);
imagesc(heteKnYX{2,3});
title('bin->con (2,3)');



%%

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