[lambda,phi] = Fu_DF_single_graph( L,  ePv1', opts);


ind = 10;


 lambda=lambdacoef;
 
 clear proto
for i = 1:10
proto(:,i) = abs(lambda).*phi(6180+i,:)';
end

test = repmat(abs(lambda),1,6180).*phi(1:6180,:)';

dist = slmetric_pw(proto,test,'sqdist');
clear id a 
[a,id]=max(dist*1000);
Sres= confusion_matrix(50, test_img_label(:),zsl_label(id(:)));    
Sres(isnan(Sres)) = 0;
acc= sum(diag(Sres))/10


%%
addpath('/import/geb-datasets/yanwei/latest_wiki_data/internal/diffusionMap');

X = ePv1';
XtX = X'*X;
data = sqrt(diag(XtX));    
%    gK{i} = X./(X_l2norm{i}*X_l2norm{i}');
data = X./repmat(data', size(X,1),1);

% construct the multiple graphs.
%for i = 1:gNum,
    % compute the similarity, the inverse of the distance in our current
    % implementation.
dist = slmetric_pw(data, data,'sqdist');
  %  gK{i} = 1./dist; %1./(1+dist);
md = median(dist(:));

test_flag=1;
if test_flag % Estimate distribution of k-nearest neighbor
    D= dist;
    D_sort = sort(D,2);
    k=30, %30;
    dist_knn = D_sort(:,1+k);  % distance to k-nn
    median_val = median(dist_knn), eps_val = median_val^2/2,
    sigmaK = sqrt(2)*median_val;
    figure, hist(dist_knn); colormap cool;
    title('Distribution of distance to k-NN');
end


eps_val=0.05;  
neigen=100;
flag_t=0; %flag_t=0 => Default: multi-scale geometry
if flag_t
    t=3;  % fixed time scale  
end

[X, eigenvals, psi, phi] = Fu_diffuse(D,eps_val,neigen);


clear proto
for i = 1:10
proto(:,i) = X(6180+i,:)';
end

test = X(1:6180,:)';
 
dist = slmetric_pw(proto,test,'nrmcorr');
clear id a 
[a,id]=max(dist*1000);
Sres= confusion_matrix(50, test_img_label(:),zsl_label(id(:)));    
Sres(isnan(Sres)) = 0;
acc= sum(diag(Sres))/10;








%%

ind = 1;
for ind=1:600
    
lambda2 = lambda(1:ind); 
phi2 = phi(:,1:ind);

 clear proto
for i = 1:10
proto(:,i) = abs(lambda2).*phi2(6180+i,:)';
end

test = repmat(abs(lambda2),1,6180).*phi2(1:6180,:)';

dist = slmetric_pw(proto,test,'sqdist');
clear id a 
[a,id]=max(dist*1000);
Sres= confusion_matrix(50, test_img_label(:),zsl_label(id(:)));    
Sres(isnan(Sres)) = 0;
acc(ind)= sum(diag(Sres))/10;

end
