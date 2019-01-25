% read .csv metadata file
% double click .csv file and choose text format for 1-3th columns and number
% format for 4th column

% input name: combined3classgt2
% No = size(combined3classgt2,1);
% idx = 1:2:No;
% 
% temp = combined3classgt2(idx,:);
% crossdb.dbtype = temp(:,1);
% crossdb.subject = temp(:,2);
% crossdb.filename = temp(:,3);
% crossdb.emotion = cell2mat(temp(:,4));

% counting
N = numel(crossdb.subject);
subjID = 0;
for i = 1:N
    subjectName = strcat(crossdb.dbtype{i},crossdb.subject{i});
    if i == 1
        subjID =  subjID + 1;
    else
        if strcmp(previousName,subjectName) == 0
            subjID =  subjID + 1;
        end
    end
    previousName = subjectName;
    crossdb.id(i,1) = subjID;
end

% save(fullfile('data','Annotation4crossdb.mat'),'crossdb');