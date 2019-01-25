% Write the data in mat file into a log text file
clear all;

% Load 'crossdb'
load(fullfile('data','Annotation4crossdb.mat'),'crossdb'); 
% Load 'preLabelsT' and 'gtLabelsT'
load(fullfile('data','results_12_15_0_13.mat'),'preLabelsT','gtLabelsT'); 


% Write file
fid = fopen('SampleLog.txt', 'wt');
% Output
idxGlobal = 1;
subIDs = unique(crossdb.id);
for i = 1:numel(subIDs)
    idx = find(crossdb.id==subIDs(i));
    fprintf(fid,'%s %s\n', crossdb.dbtype{idx(1)},crossdb.subject{idx(1)});
    for j = 1:numel(idx)
        fprintf(fid,'%s %d %d\n', crossdb.filename{idx(j)},gtLabelsT(idxGlobal)-1,preLabelsT(idxGlobal)-1);
        idxGlobal = idxGlobal + 1;
    end
end
fclose(fid);  