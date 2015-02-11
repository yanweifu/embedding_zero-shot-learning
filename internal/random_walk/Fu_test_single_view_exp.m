% data explanations:
zsl_label = zsl_label(:);
test_img_label=test_img_label(:);

% prototype:
lowPro2 = Pconzsl_nrm_bin_q_low.max_new_conf_prototype;
binPro2; 
conPro2;
%  extended prototype:
ePv1 =[Pv1; lowPro2]; ePv2 =[Pv2; conPro2]; ePv3 = [Pv3;binPro2];
no = size(Pv1,1);

%  extended prototype:
No_test_cls = 10;

% do it with two views:
vdata{1} = ePv1';
vdata{2} = ePv2';
vdata{3} = ePv3';

gamma = 0.5; %0.85;
L = [zeros(no,No_test_cls);  2*eye(No_test_cls)-1]; 
%L = [(1-gamma).*dist';  2*eye(No_test_cls)-1]; 

opts =struct();
[sLabels, cLabels, sL1Labels, cL1Labels,cZmat] = Fu_mvsslL2_v2( L,  vdata, opts);

for i = 1:3
    zsLabels{i} =zsl_label(sLabels{i});
    zsL1Labels{i} = zsl_label(sL1Labels{i});    
end

fprintf('L2-SSL results:\n');
%show_result(sLabels, cLabels, test_label, no);
show_result(zsLabels, zsl_label(cLabels), test_img_label);

% combining them together:
L2 = L;
clear acc_sCombine
for itr = 1:20
    % clamping:
   % L2(6181:end,:) = [2*eye(No_test_cls)-1];
    
    L2= (1-alpha)*cZmat*L2;
    [~, sLabels] = max(L2(1:no,:), [], 2);
    Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_sCombine(itr)= sum(diag(Sres))/10;
end



%% low-level feature view:
data = ePv1';
dist = slmetric_pw(lowPro2', Pv1','nrmcorr');
[init,id]= max(dist);
Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(id(:)));    
Sres(isnan(Sres)) = 0;
fprintf('low-level feature (CCA) space accuracy: %f',sum(diag(Sres))/10);
 
 
addpath('/homes/yf300/Researchfile/Lib/MATLAB_util/');
subplot(221)
ccDrawConfMat(Sres(zsl_label,zsl_label)); 

clear acc_single_conf2 acc_single_conf
  
gamma = 0.5; %0.85;
%L = [zeros(no,No_test_cls);  2*eye(No_test_cls)-1]; 
L = [(1-gamma).*dist';  2*eye(No_test_cls)-1]; 

opts =struct();
 [sLabels, Zmat, sL1Labels] = Fu_singlesslL2_v2( L,  data, opts);

 % accuracy:
maxcls = max(test_img_label);
Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
Sres(isnan(Sres)) = 0;
acc_sLow(i)= sum(diag(Sres))/10;

Sresl1= confusion_matrix(maxcls, test_img_label(:),zsl_label(sL1Labels));    
Sresl1(isnan(Sresl1)) = 0;
accl1_sLow(i)= sum(diag(Sresl1))/10;     

subplot(222)
allres = Sres(zsl_label,zsl_label);
ccDrawConfMat(allres);

% do more iteration to update the results:
L2 = L;
for itr = 1:20
        % clamping:
%L2(6181:end,:) = [2*eye(No_test_cls)-1];
    
L2= (1-alpha)*Zmat*L2;
[~, sLabels] = max(L2(1:no,:), [], 2);
    Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_sLow(itr)= sum(diag(Sres))/10;
end



%% single comparison of view2 (word space):

clear acc_single_conf2 acc_single_conf
data = ePv2';

dist = slmetric_pw(conPro2', Pv2','nrmcorr');
[a,id]= max(dist);
Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(id(:)));    
Sres(isnan(Sres)) = 0;
 sum(diag(Sres))/10
 
 cset = [0];
L = [zeros(no,No_test_cls);  2*eye(No_test_cls)-1]; 

for i = 1:length(cset)
    opts.c= exp(cset(i));
     [sLabels, Zmat, sL1Labels] = Fu_singlesslL2_v2( L,  data, opts);
         
     maxcls = max(test_img_label);
    Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_single_conf(i)= sum(diag(Sres))/10;
    
    Sresl1= confusion_matrix(maxcls, test_img_label(:),zsl_label(sL1Labels));    
    Sresl1(isnan(Sresl1)) = 0;
    accl1(i)= sum(diag(Sresl1))/10;     
    
L2 = L;
for itr = 1:20
L2= (1-alpha)*Zmat*L2;
[~, sLabels] = max(L2(1:no,:), [], 2);
    Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_single_conf2(i, itr)= sum(diag(Sres))/10;
end
end



%% single comparison of attribute views:

clear acc_single_conf2 acc_single_conf
data = ePv3';

dist = slmetric_pw(binPro2', Pv3','nrmcorr');
[a,id]= max(dist);
Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(id(:)));    
Sres(isnan(Sres)) = 0;
 sum(diag(Sres))/10
 
 cset = [0];
L = [zeros(no,No_test_cls);  2*eye(No_test_cls)-1]; 

for i = 1:length(cset)
    opts.c= exp(cset(i));
     [sLabels, Zmat, sL1Labels] = Fu_singlesslL2_v2( L,  data, opts);
         
     maxcls = max(test_img_label);
    Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_single_conf(i)= sum(diag(Sres))/10;
    
    Sresl1= confusion_matrix(maxcls, test_img_label(:),zsl_label(sL1Labels));    
    Sresl1(isnan(Sresl1)) = 0;
    accl1(i)= sum(diag(Sresl1))/10;     
    
L2 = L;
for itr = 1:20
L2= (1-alpha)*Zmat*L2;
[~, sLabels] = max(L2(1:no,:), [], 2);
    Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_single_conf2(i, itr)= sum(diag(Sres))/10;
end
end
%%  Diffusion metric:



clear acc_single_conf2 acc_single_conf
data = ePv3';

% dist = slmetric_pw(binPro2', Pv3','nrmcorr');
% [a,id]= max(dist);
% Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(id(:)));    
% Sres(isnan(Sres)) = 0;
%  sum(diag(Sres))/10
 
 cset = [0];
L = [zeros(no,No_test_cls);  2*eye(No_test_cls)-1]; 


[lambda,phi] = Fu_DF_single_graph( L,  data, opts);

for i = 1:10
proto(:,i) = lambda.^2.*phi(6180+i,:)';
end

test = repmat(lambda.^2,1,6180).*phi(1:6180,:)';

dist = slmetric_pw(proto,test,'sqdist');
[a,id]=max(dist);
 Sres= confusion_matrix(50, test_img_label(:),zsl_label(id(:)));    
    Sres(isnan(Sres)) = 0;
    acc= sum(diag(Sres))/10

     


     

for i = 1:length(cset)
    opts.c= exp(cset(i));
     [sLabels, Zmat, sL1Labels] = Fu_singlesslL2_v2( L,  data, opts);
         
     maxcls = max(test_img_label);
    Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_single_conf(i)= sum(diag(Sres))/10;
    
    Sresl1= confusion_matrix(maxcls, test_img_label(:),zsl_label(sL1Labels));    
    Sresl1(isnan(Sresl1)) = 0;
    accl1(i)= sum(diag(Sresl1))/10;     
    
L2 = L;
for itr = 1:20
L2= (1-alpha)*Zmat*L2;
[~, sLabels] = max(L2(1:no,:), [], 2);
    Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_single_conf2(i, itr)= sum(diag(Sres))/10;
end
end


