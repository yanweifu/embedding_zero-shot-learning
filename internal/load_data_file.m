function load_data_file(datapath,saveasname,len)
%load_data_file(datapath,saveasname,len)
%
if nargin<3
    len =300;
elseif nargin<2
    saveasname='data_svr_res.mat';
elseif nargin<1
    datapath='.';
end
%datapath=['./data/'];
file = dir([datapath,'data_*.mat']);
tmp =load([datapath, file(1).name]);

all_pL_Xte= zeros(length(tmp.pL_Xte),len);
all_pL_Xtr = zeros(length(tmp.pL_Xtr),len);
for i=1:length(file)
    p1=strfind(file(i).name,'_');
    p2=strfind(file(i).name,'.');

    load([datapath,file(i).name]);  
    idx =str2num(file(i).name(p1+1:p2-1));
  
       all_pL_Xte(:,idx)=pL_Xte;
    all_pL_Xtr(:,idx)=pL_Xtr;    
end

%save('AwA_clsname_v2_pL_Xte_Xtr.mat','all_pL_Xte','all_pL_Xtr')
%save('AwA_clsname_v3_300dim_pL_Xte_Xtr.mat','all_pL_Xte','all_pL_Xtr')
save(saveasname,'all_pL_Xte','all_pL_Xtr');

