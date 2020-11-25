clear all; clc;
load integers_dlm.mat;
dag = OutputStructure_1997;
para = dlm_basic_eval_updated(dlm, dag,1);
data = load('dlm_resultsFinalRun1997.txt');

aic = aic(data(:,4),data(:,3), para);
bic = bic(data(:,4),data(:,3), para);