% Extract optical flow for image sequences
clear all;

% Setups
% rows = 300;
% cols = 245;
rows = 220;
cols = 200;
alpha = 5:12;

metadataPath = fullfile('..','data','Annotation4crossdb.mat');

% Load data
load(metadataPath,'crossdb'); % database name: crossdb
% Extracting
Nseq = numel(crossdb.subject);
fprintf('\nOptical Flow of sequences are generating...\n\n');
for numAug = 1:numel(alpha)
    fprintf('*************************\n');
    fprintf('alpha = %d\n',alpha(numAug));
    fprintf('*************************\n');
    rootDir = fullfile('..','dataset', ['MEGC2019_alpha' num2str(alpha(numAug))]);
    savePath = fullfile('..','data', ['crossdb_mat_' num2str(alpha(numAug)) '.mat']);
    % Process each sequence
    imdb.data = [];
    imdb.labels = [];
    imdb.id = [];
    for i = 1:Nseq
        seqPath = fullfile(rootDir,crossdb.dbtype{i},crossdb.subject{i},crossdb.filename{i});
        if strcmp(crossdb.dbtype{i},'smic')
            filePostfix = '*.bmp';
        else
            filePostfix = '*.jpg';
        end
        fileList = dir(fullfile(seqPath,filePostfix));
        [idxApex,idxOnset] = detect_apex_frame(fileList);
        filePathOnset = fullfile(fileList(idxOnset).folder,fileList(idxOnset).name);
        filePathApex = fullfile(fileList(idxApex).folder,fileList(idxApex).name);
        img1 = imresize(imread(filePathOnset),[rows cols]);
        img2 = imresize(imread(filePathApex),[rows cols]);
        
        fprintf('The %d-th sequence...\n',i);
        % extract the optimal flow
        uv = estimate_flow_interface(img1, img2, 'classic+nl-fast');
        mo(:,:,1) = sqrt(uv(:,:,1).^2+uv(:,:,2).^2);
        mo(:,:,2) = atan(uv(:,:,1)./uv(:,:,2));
        imdb.data = cat(4,imdb.data,cat(3,uv,mo));
    end
    imdb.labels = crossdb.emotion;
    imdb.id = crossdb.id;
    % save generated database
    save(savePath,'imdb');
end