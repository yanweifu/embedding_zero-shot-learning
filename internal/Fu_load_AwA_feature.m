function     Fu_load_AwA_feature(folderT,save_mat_file_name,feature_name)

subfolder = dir([folderT,'*']);

i=1;

data = cell(1,length(subfolder));
while i<=length(subfolder)
    if ~subfolder(i).isdir || strcmp(subfolder(i).name,'.')|| strcmp(subfolder(i).name,'..')||strcmp(subfolder(i).name,'mat')
        i = i+1;
        continue;
    end
    path1 = [folderT, subfolder(i).name,'/'];
    
    
    files = dir([path1,'*.txt']);
    
    v1 = genvarname(feature_name);
    for f1 = 1:length(files)
        pth=[path1,files(f1).name];
        eval(['files(f1).',v1,'= importdata(pth);']);
    end
    
    data{i}.name = subfolder(i).name;
    data{i}.data = files;
    
    i = i+1;
end
save(save_mat_file_name,'data');


