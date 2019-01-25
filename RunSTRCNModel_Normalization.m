% Train the deep model of RCNN by subjects
clear all;

% Settings
sizeType = 'size220';
imageSize = [220 200];
NitersInterEpoch = 0;%1,2,3,5,10,15
maps = 128; % 128, 256, 512, 600, 1024
alpha = 5:12;
alpha_te = 8;
lr = 1e-3;%1e-2
ratio = 0.8;
metaPath = fullfile('data','Annotation4crossdb.mat');

% -------------------------------------------------------------------------
%                              Model Trainig                                           
% -------------------------------------------------------------------------
% Load all data into memory for training, size: about 3.0G
imdb_tr.data = [];
imdb_tr.labels = [];
imdb_tr.id = [];
for numAug = 1:numel(alpha)
    imdbPath = fullfile('data',sizeType,['crossdb_mat_' num2str(alpha(numAug)) '.mat']);
    tmpImdb = load(imdbPath); % load 'imdb'
    % normalize data
%     tmpImdb.imdb.data = normalizeSample(tmpImdb.imdb.data);
%     tmpImdb.imdb.data = normalizeSubject(tmpImdb.imdb.data,tmpImdb.imdb.id);
    tmpImdb.imdb.data = normalizeDataset(tmpImdb.imdb.data); % normalize data
    imdb_tr.data = cat(4,imdb_tr.data,tmpImdb.imdb.data); % cascade data
    imdb_tr.labels = cat(1,imdb_tr.labels,tmpImdb.imdb.labels+1); % cascade label
    imdb_tr.id =  cat(1,imdb_tr.id,tmpImdb.imdb.id);
end
load(metaPath); % load 'crossdb'
% Load the testing data
imdbPath = fullfile('data',sizeType,['crossdb_mat_' num2str(alpha_te) '.mat']);
load(imdbPath); % load 'imdb'
% imdb.data = normalizeSample(imdb.data);
% imdb.data = normalizeSubject(imdb.data,imdb.id); % normalize data
imdb.data = normalizeDataset(imdb.data); % normalize data
imdb.labels = imdb.labels + 1;
% 
subIdx = unique(crossdb.id);

preLabelsT = [];
gtLabelsT = [];
% each subject
tic;
for i = 1:numel(subIdx)
    % model training
%     net = strcnInit(imageSize);
%     net = strcnInit_balanced(imageSize);
    net = strcnInit_deeper(imageSize,maps);
    
    index_te = find(imdb_tr.id == subIdx(i));
    Ns = numel(imdb_tr.labels);
    set = 1:Ns;
    index_tr = setdiff(set,index_te);
    
    fprintf('The trainning begin...\n');
    net = rcnnTrain(net,imdb_tr.data(:,:,:,index_tr),imdb_tr.labels(index_tr),NitersInterEpoch,lr);
    netStruct = net.saveobj();
    modelPath = fullfile('model',['strcn_obj_' num2str(i) '.mat']);
    save(modelPath,'netStruct','-v7.3');
    fprintf('The trainning done...\n');
    
    % model training
    % load(fullfile('data','strcn-dag-12_12_13_4.mat'),'netStruct');
    % net = dagnn.DagNN.loadobj(netStruct);
    fprintf('Testing...\n');
    index_te = find(imdb.id == subIdx(i));
    [perf,Labels] = rcnnTest(net,imdb.data(:,:,:,index_te),imdb.labels(index_te));
    confMatrix = calConfusionMatrix(double(Labels),imdb.labels(index_te), unique(imdb.labels));
    fprintf('Accuracy is %.3f.\n',perf);
    
    perfT(i) = perf;
    ConfMatrixT(:,:,i) = confMatrix;
    preLabelsT = cat(1,preLabelsT,Labels);
    gtLabelsT = cat(1,gtLabelsT,imdb.labels(index_te));
end
% Evaluate the performance
elapsedTime = toc;
elapsedTime = elapsedTime/3600;
[avgAcc,Acc] = calBalancedAccuracy(preLabelsT,gtLabelsT);
[avgFscore,Fscores] = calF1Score(preLabelsT,gtLabelsT);
%% Save results
currenttime = datevec(now);
timestamp = [ num2str(currenttime(2)) '_' num2str(currenttime(3)) '_' num2str(currenttime(4)) '_' num2str(currenttime(5))];
fprintf('The model has been saved.\n');
save(fullfile('data',['results_' timestamp '.mat']),'avgAcc','Acc','avgFscore','Fscores','preLabelsT','gtLabelsT');