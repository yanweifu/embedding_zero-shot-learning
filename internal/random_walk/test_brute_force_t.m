% 
% tset=1:4;
% de = 0.001;
% eset =0.05:0.02:0.2;
% clear acc
% 
% for ti = 1:length(tset)
% for eid =1:length(eset)
%     
%     opts.e = eset(eid);
%     opts.t = tset(ti);
%     opts.delta = de;
% [lambda,phi] = Fu_DF_single_graph( L,  ePv1', opts);
% 
% 
% for ind=1:2:2000
% lambda2 = lambda(1:ind); 
% phi2 = phi(:,1:ind);
% 
%  clear proto
% for i = 1:10
% proto(:,i) = abs(lambda2).*phi2(6180+i,:)';
% end
% 
% test = repmat(abs(lambda2),1,6180).*phi2(1:6180,:)';
% 
% dist = slmetric_pw(proto,test,'sqdist');
% clear id a 
% [a,id]=max(dist*1000);
% Sres= confusion_matrix(50, test_img_label(:),zsl_label(id(:)));    
% Sres(isnan(Sres)) = 0;
% acc(ti,ind,eid)= sum(diag(Sres))/10;
% end
% 
% end
% end


%eps_val=0.05;  
neigen=300;
eset =[0.2:0.05:0.4];
for eid=1:length(eset)
    for tid = 1:5
        t = tid;
        eps_val = eset(eid);

[X, eigenvals, psi, phi] = Fu_diffuse(D,eps_val,neigen,t);


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
acc(eid,tid)= sum(diag(Sres))/10;
    end
end