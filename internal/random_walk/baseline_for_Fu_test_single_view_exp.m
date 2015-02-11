data = [ePv1'; ePv2'; ePv3'];


dist3 = slmetric_pw([lowPro2'; conPro2'; binPro2'], [Pv1'; Pv2';Pv3'],'nrmcorr');
[init,id]= max(dist3);
Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(id(:)));    
Sres(isnan(Sres)) = 0;
fprintf('low-level feature (CCA) space accuracy: %f',sum(diag(Sres))/10);
addpath('/homes/yf300/Researchfile/Lib/MATLAB_util/');
subplot(221)
ccDrawConfMat(Sres(zsl_label,zsl_label)); 
title('direct concate 3 view');



dist2 = slmetric_pw([conPro2';binPro2'], [Pv2'; Pv3'],'nrmcorr');
[init,id]= max(dist2);
Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(id(:)));    
Sres(isnan(Sres)) = 0;
fprintf('low-level feature (CCA) space accuracy: %f',sum(diag(Sres))/10);
subplot(222)
ccDrawConfMat(Sres(zsl_label,zsl_label)); 
title('direct concate 2 view');



 


clear acc_single_conf2 acc_single_conf
  
gamma = 0.5; %0.85;
L = [zeros(no,No_test_cls);  2*eye(No_test_cls)-1]; 
%L = [(1-gamma).*dist';  2*eye(No_test_cls)-1]; 

opts =struct();
 [sLabels, Zmat, sL1Labels] = Fu_singlesslL2_v2( L,  data, opts);

 % accuracy:
maxcls = max(test_img_label);
Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
Sres(isnan(Sres)) = 0;
acc_sLow3(i)= sum(diag(Sres))/10;

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
    acc_sLow3(itr)= sum(diag(Sres))/10;
end