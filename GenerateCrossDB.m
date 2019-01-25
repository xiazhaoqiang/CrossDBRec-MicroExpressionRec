% Create a cross-dataset based on the metadata and individual datasets
clear all;

load(fullfile('data','Annotation4crossdb_v1.0.mat'),'crossdb');

saveRootDir = 'E:\Datasets\MEGC2019';
readRootDir_casme2 = 'E:\Datasets\CASME2\ProcessedData\Cropped';
readRootDir_smic = 'E:\Datasets\SMIC\ProcessedData\SMIC_all_cropped\HS';
readRootDir_samm = 'E:\Datasets\SAMM\ProcessedData\Cropped';

Nsamples = numel(crossdb.subject);
fprintf('Cross dataset is generating...\n');
for i = 1:Nsamples
    crossdb.dbtype{i} = convertStringsToChars(crossdb.dbtype{i});
    crossdb.subject{i} = convertStringsToChars(crossdb.subject{i});
    crossdb.filename{i} = convertStringsToChars(crossdb.filename{i});
    switch crossdb.dbtype{i}
        case 'casme2' % 1-145
            readPath = fullfile(readRootDir_casme2,crossdb.subject{i},crossdb.filename{i});
        case 'smic' % 146-309
            if crossdb.emotion(i) == 0
                type = 'negative';
            elseif crossdb.emotion(i) == 1
                type = 'positive';
            else
                type = 'surprise';
            end
            readPath = fullfile(readRootDir_smic,crossdb.subject{i},'micro',type,crossdb.filename{i});
        case 'samm' % 310-441
            crossdb.subject{i} = sprintf('%03s',crossdb.subject{i});
            readPath = fullfile(readRootDir_samm,crossdb.subject{i},crossdb.filename{i});
        otherwise
            warning('Unexpected database type. No database created.')
            continue;
    end
    
    % generate the paths
    savePath = fullfile(saveRootDir,crossdb.dbtype{i});
    if ~exist(savePath)
        mkdir(savePath);
    end
    savePath = fullfile(savePath,crossdb.subject{i});
    if ~exist(savePath)
        mkdir(savePath);
    end
    savePath = fullfile(savePath,crossdb.filename{i});
    if ~exist(savePath)
        mkdir(savePath);
    end
    % copy file from source dir to target dir
    copyfile(readPath,savePath);
    fprintf('The %d-th sequence.\n',i);
end
fprintf('Cross dataset has been generated...\n');
save(fullfile('data','Annotation4crossdb.mat'),'crossdb');