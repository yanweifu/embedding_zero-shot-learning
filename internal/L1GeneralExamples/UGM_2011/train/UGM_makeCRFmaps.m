function [nodeMap,edgeMap] = UGM_makeCRFmaps(Xnode,Xedge,edgeStruct,ising,tied,paramLastState)
% Assumes that all nodes have the same number of states

if nargin < 6
	paramLastState = 1;
end

nNodes = size(Xnode,3);
nEdges = edgeStruct.nEdges;
nStates = edgeStruct.nStates;
maxState = max(nStates);

UGM_assert(min(nStates)==maxState,'UGM_makeCRFMaps assumes that all nodes must have the same number of states');
nStates = nStates(1);

nNodeFeatures = size(Xnode,2);
nEdgeFeatures = size(Xedge,2);

nodeMap = zeros(nNodes,nStates,nNodeFeatures,'int32');
if tied
	fNum = 1;
	for f = 1:nNodeFeatures
		if paramLastState
			for s = 1:nStates
				nodeMap(:,s,f) = fNum;
				fNum = fNum+1;
			end
		else
			for s = 1:nStates-1
				nodeMap(:,s,f) = fNum;
				fNum = fNum+1;
			end
		end
	end
else
	nodeMap(:) = 1:numel(nodeMap);
end
nNodeParams = max(nodeMap(:));

edgeMap = zeros(nStates,nStates,nEdges,nEdgeFeatures,'int32');
if tied
	switch ising
		case 1
			for f = 1:nEdgeFeatures
				for s = 1:nStates
					edgeMap(s,s,:,f) = nNodeParams+f;
				end
			end
		case 2
			fs = 1;
			for f = 1:nEdgeFeatures
				for s = 1:nStates
					edgeMap(s,s,:,f) = nNodeParams+fs;
					fs = fs+1;
				end
			end
		case 0
			ssf = 1;
			for f = 1:nEdgeFeatures
				for s1 = 1:nStates
					for s2 = 1:nStates
						edgeMap(s1,s2,:,f) = nNodeParams+ssf;
						ssf = ssf+1;
					end
				end
			end
	end
else
	switch ising
		case 1
			fe = 1;
			for f = 1:nEdgeFeatures
				for e = 1:nEdges
					for s = 1:nStates
						edgeMap(s,s,e,f) = nNodeParams+fe;
					end
					fe = fe+1;
				end
			end
		case 2
			fse = 1;
			for f = 1:nEdgeFeatures
				for e = 1:nEdges
					for s = 1:nStates
						edgeMap(s,s,e,f) = nNodeParams+fse;
						fse = fse+1;
					end
				end
			end
		case 0
			fsse = 1;
			for f = 1:nEdgeFeatures
				for e = 1:nEdges
					for s1 = 1:nStates
						for s2 = 1:nStates
							edgeMap(s1,s2,e,f) = nNodeParams+fsse;
							fsse = fsse+1;
						end
					end
				end
			end
	end
end