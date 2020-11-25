 
clc;clear all;close all;
%% load data
load integers_dlm.mat;
filename = 'eval_results1.txt';
 
results = fopen(filename,'w');
i = 1;
FuncName = ['OutputStructure_1_',num2str(i)];
 
    %% Evaluate DAG
 
    para = dlm_basic_eval(dlm,feval(FuncName),i);
    data = load(['dlm_resultsFinalRun',num2str(i),'.txt']);
 
    %% Record Results
    aic = aic(data(:,5),data(:,4), para);
    bic = bic(data(:,5),data(:,4), para);
 
    aucR(i) = auc(data(:,4),data(:,5));
    rmseR(i) = sqrt(mean((data(:,4)-data(:,5)).^2));
    accuracyR(i) =  mean(round(data(:,4))==data(:,5));
    
    fprintf(results, '%d %.5f %.5f %.5f %.5f %.5f\n',i,aucR(i),rmseR(i),accuracyR(i), aic, bic)
   
 
fclose(results);
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
