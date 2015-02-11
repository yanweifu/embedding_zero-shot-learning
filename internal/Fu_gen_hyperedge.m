function [acc,hyperedge,meta] =Fu_gen_hyperedge(vdata, opts,test_img_label,zsl_label)
%
% input: vdata: 3 modality: 
%
% return: hyperedge: double zscored hyperedge. (No.edge * Node index);
%
%  parameters need to be decided:
%               opts.k:  the nearest neighbour parameters to decide the similarity matrix of data points from the same view.
%                               In our ECCV submission, this parameter is very robust (k=10~50);
%
%               opts.prototypeTime: the prototypeTime of graph in each view. Special for labelled/prototypes nodes:
%
%               opts.hypk: the nearest neighbour parameters for hypergraphs. To decide the size of hyperedges.
%               opts.hyperProto:  the prototypeTime of hypergraph. Specially for labelled/prototype nodes;
%
%               opts.visualization: if want to visualization, turn it on; and I will output the similarity matrix of intermediate results of cross-view
%               and each view.
%
%               opts.WeightPrior_eachview:  ([1,1.5,1] defaulted), the prior weight of each view. Sometimes we always have some
%               prior knowledge on how strong of each view. 
%
%               opts.Weight_edge_vs_hyperedge: ([0.25,0.75] defaulted). How to combined the similarity of each edge with the
%               similarity of hyperedges.
%                 
%               opts.no, the number of testing instances. default: AwA, 6180.
%

opts = getPrmDflt(opts,{ 'k',30,'hypk',30, 'alpha',0.5,'no',6180,'hyperedge',[],'hyperProto',9,'prototypeTime', 9, 'visualization',0,'WeightPrior_eachview',[1,1.5,1],'Weight_edge_vs_hyperedge',[0.25,0.75]},-1);
k = opts.k; alpha = opts.alpha; no = opts.no;

% if we want to do visualization, we must repeatedly compute the hyperedge:
if opts.visualization
    opts.hyperedge =[];
end

gNum = length(vdata);

gK = cell(gNum, 1);
dN = size(vdata{1},2);

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
for i=1:gNum
    gKnn{i} = filterKnn(gK{i},opts);
    zsgKnn{i} = filterKnn(zsgK{i},opts);    
end

 %w=[1, 1.5, 1];
w =opts.WeightPrior_eachview;

% results of combinations (no-zscore);
if gNum==3
PG= combineWeight(vdata);
else
    PG=ones(length(w),1);
end
cW = zeros(dN, dN);
 for i = 1:gNum,
    cW = cW + w(i)*PG(i)*gKnn{i};
end
accgKnn = db_cW(cW,test_img_label,zsl_label,dN,opts);

meta.accgKnn = accgKnn;

if opts.visualization
    for i=1:gNum
        accgKnnXX{i} = db_cW(gKnn{i}, test_img_label,zsl_label,dN,opts);
    end
    meta.accgKnnXX= accgKnnXX;
    meta.gKnn =gKnn;
    meta.zsgKnn = zsgKnn;
end

%% compute the hyperedge:
%check whether the hyperedge is precomputed:
if isempty(opts.hyperedge)
    opthyper.k = opts.hypk;     opthyper.prototypeTime = opts.hyperProto;     opthyper.no = opts.no;

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
           zsgKnnXY{i,j} = filterKnn(zsgXY{i,j},opthyper);
           if opts.visualization
                   gKnnXY{i,j} = filterKnn(gKXY{i,j},opts);
           end
            % then using view Y to query view X: 
           hdistYX = slmetric_pw(Y, X,'sqdist');
           md = median(hdistYX(:));
           gKYX{i,j}= exp(-hdistYX./(md));

           zsgYX{i,j} = doMatrxZscore(gKYX{i,j});

           % colect the k-nearest neighour
           zsgKnnYX{i,j} = filterKnn(zsgYX{i,j},opthyper);
           
           if opts.visualization
                gKnnYX{i,j} = filterKnn(gKYX{i,j},opts);
                  accgKnnXY{i,j} = db_cW(gKnnXY{i,j},test_img_label,zsl_label,dN,opts);
                  accgKnnYX{i,j} = db_cW(gKnnYX{i,j},test_img_label,zsl_label,dN,opts);
           end
           
           % results of each view:
          acczsgKnnYX{i,j}=db_cW(zsgKnnYX{i,j},test_img_label,zsl_label,dN,opts);
          acczsgKnnXY{i,j}=db_cW(zsgKnnXY{i,j},test_img_label,zsl_label,dN,opts);       
        end
    end

    meta.acczsgKnnYX =acczsgKnnYX;
    meta.acczsgKnnXY =acczsgKnnXY;
   
   if opts.visualization
       meta.zsgKnnYX =zsgKnnYX;
       meta.zsgYX = zsgYX;
       
       meta.zsgKnnXY =zsgKnnXY;
       meta.zsgYX =zsgYX;
       
       meta.gKnnXY =gKnnXY;
       meta.gKXY = gKXY;
       meta.gKnnYX =gKnnYX;
       meta.gKYX = gKYX;
       
       meta.accgKnnXY = accgKnnXY;
       meta.accgKnnYX = accgKnnYX;
   end
   
   
    % get zscored hyperedges:
    hyperedge =[];
    for i=1:gNum
        for j=i+1:gNum
            hyperedge =[hyperedge; zsgKnnXY{i,j};zsgKnnYX{i,j}];
        end
    end
else
    % if the hyperedge is precomputed, directly assign it.
    hyperedge =opts.hyperedge;
end

%%
% combine hyperedge into new similarity matrix:
nodeidx=1:size(hyperedge,2);
simhyper= Knn_hypergraph_sim(hyperedge, nodeidx);
meta.hyperSim =simhyper;

% do the testing for the hyperedge's similarity matrix:
accHyperedge = db_cW(simhyper,test_img_label,zsl_label,dN,opts);
meta.accHyperedge= accHyperedge;

%% combine each individual view and hyperedge's similarity:
%wxy=[0.3448    0.6552];
%wxy =[0.25,0.75];
wxy = opts.Weight_edge_vs_hyperedge;
cw_Comb = cW*wxy(1) + simhyper*wxy(2);
acc = db_cW(cw_Comb,test_img_label,zsl_label,dN,opts);

meta.cw_Comb = cw_Comb;



