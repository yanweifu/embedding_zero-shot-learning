function [acc_sCombine,labels ]=db_cW(cW,test_img_label,zsl_label,dN,opts)
% for AwA data:
%no = 6180;No_test_cls = 10;
no = opts.no; No_test_cls = length(zsl_label);  maxcls =max(zsl_label);

L = [zeros(no,No_test_cls);  2*eye(No_test_cls)-1];  



alpha =0.35; 
D = 1./sqrt(sum(cW,2));
D = sparse(diag(D));
LAP = D*sparse(cW)*D;

Zmat = inv(eye(dN)-alpha*LAP); 
cLabels = (1-alpha)*Zmat*L;



% combining them together:
L2 = L;
clear acc_sCombine
for itr = 1:20
    % clamping:
    %L2(6181:end,:) = [2*eye(No_test_cls)-1];
    
    L2= (1-alpha)*Zmat*L2;
    [~, sLabels] = max(L2(1:no,:), [], 2);
    Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_sCombine(itr)= sum(diag(Sres))/No_test_cls;
    res{itr} = sLabels;
end

[~,id]=max(acc_sCombine);
labels = res{id};