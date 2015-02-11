function sim = Knn_hypergraph_sim(hypergph, nodeidx)
% hypergph: each row is corresponding to a hyperedge, connecting to all nodes (column);
%                   hypergph: No.edge *No.node;
%
% %% debug:
% hypergph = hyp; 
% nodeidx = [1:6180];

%% generate the incidence matrix:
% incidence: No. nodes *No. edges:
incidence = hypergph(:,nodeidx)>0;
incidence = incidence';
%%  softly assign the weight of each vertex to hyperedge:
% delta_t in the paper.
hypedgeweight = sum(hypergph,2)./sum(incidence,1)';

% to compute hn: No.node *No.edge: 
nonode = length(nodeidx); Noedge= size(incidence,2);
% distribution of each node on each edges. distrNodeOnEdges: No. edge *No.node:
distrNodeOnEdges = repmat(hypedgeweight,1,nonode).*incidence'.*hypergph(:,nodeidx).^2;
sumdistrNodeOnEdges = sum(distrNodeOnEdges,1);
% sum weights of each node: sumeachNodeOnAllEdges: No. nodes;
sumeachNodeOnAllEdges= sqrt(sumdistrNodeOnEdges);
% No.edges*No.node;
hn = hypergph(:,nodeidx).*incidence'./repmat(sumeachNodeOnAllEdges,Noedge,1);

%%
% equation (7) in paper:
sim = (repmat(hypedgeweight,1,nonode).*hn)'*hn;

