function [PG,temp_v12,temp_v23,temp_v13]= combineWeight(vdata)
% debug function, vdata should be 3*1 cell.
% each cell of vdata, should be No_dim *No_node
%
% Note that each instance of vdata must be l2-normalized!!!
%

opt =struct(); opt =getPrmDflt(opt,{'options','nooutlier'},-1);
    
opt.options = 'outlier';
        num = size(vdata{1},2);  
if strcmp(opt.options,'nooutlier')
  
        % combine into a single graph:

        %  the distance between the same nodes from graph 1 and graph 2.
    temp_v12 = 2*ones(num,1)  - 2*sum(vdata{1}'.*vdata{2}',2);  
        
        
        
        % we use t-student distribution:
        sim_v12 = sum(1./sqrt(temp_v12));

        temp_v23 = 2*ones(num,1) - 2*sum(vdata{2}'.*vdata{3}',2);
        sim_v23 = sum(1./sqrt(temp_v23)); 

        temp_v13 = 2*ones(num,1) - 2*sum(vdata{1}'.*vdata{3}',2);
        sim_v13 = sum(1./sqrt(temp_v13)); 


        v1_degree = sim_v12 + sim_v13;
        v2_degree = sim_v12 + sim_v23;
        v3_degree = sim_v13 + sim_v23;
        vol_v123 = v1_degree + v2_degree + v3_degree;

        % the graph priors (each degree devides the total volumn)
        P_v1 = v1_degree/vol_v123;
        P_v2 = v2_degree/vol_v123;
        P_v3 = v3_degree/vol_v123;
        PG = [P_v1, P_v2, P_v3];
elseif strcmp(opt.options, 'outlier')

        %  the distance between the same nodes from graph 1 and graph 2.
        temp_v12 = 2*ones(num,1)  - 2*sum(vdata{1}'.*vdata{2}',2);  
        % we use t-student distribution:
        thr = 1/sqrt(2);
        %thr =exp(-2);
        
        t12= 1./sqrt(temp_v12); f12 = t12>thr;
        %t12= exp(-temp_v12); f12 = t12>thr;
        sim_v12 = sum(t12.*f12);

        temp_v23 = 2*ones(num,1) - 2*sum(vdata{2}'.*vdata{3}',2);
        t23 = 1./sqrt(temp_v23); f23 = t23>thr;
        %t23 = exp(-temp_v23); f23 = t23>thr;
        sim_v23 = sum(t23.*f23); 

        temp_v13 = 2*ones(num,1) - 2*sum(vdata{1}'.*vdata{3}',2);
         t13= 1./sqrt(temp_v13); f13 = t13>thr;
        %t13= exp(-temp_v13); f13 = t13>thr;
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
        
end