clc;clear all;close all;
%% load data
load integers_dlm.mat;
filename = 'eval_results.txt';
results = fopen(filename,'w');
for i=1:2
    data = load(['dlm_resultsFinalRun',num2str(i),'.txt']);
    %% Record Results
    aucR(i) = auc(data(:,4),data(:,5));
    rmseR(i) = sqrt(mean((data(:,4)-data(:,5)).^2));
    accuracyR(i) =  mean(round(data(:,4))==data(:,5));
    cbdR(i) = cbd(data(:,5),data(:,4));
    fprintf(results, '%d %.5f %.5f %.5f %.5f\n',i,aucR(i),rmseR(i),accuracyR(i), cbdR(i));
end
fclose(results);