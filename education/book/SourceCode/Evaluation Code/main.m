%% main
clc;clear all;close all;
%% load data
load integers_dlm.mat;
%%
%dag(:,:,1) = get_integers_dlm_dag_1; 
%dag(:,:,2) = get_integers_dlm_dag_2;
%dag(:,:,3) = get_integers_dlm_dag_3;
%dag2(:,:,4) = get_integers_dlm_dag_4;
%%
%dag2(:,:,192) = structure_192;

dag1336 = OutputStructure_8_1336;
dag1338 = OutputStructure_8_1338;
dag1340 = OutputStructure_8_1340;
dag1350 = OutputStructure_8_1350;


dag4774 = OutputStructure_9_4774;
dag4782 = OutputStructure_9_4782;
dag4793 = OutputStructure_9_4793;
dag4804 = OutputStructure_9_4804;

dag4647 = OutputStructure_10_4647;
dag4651 = OutputStructure_10_4651;
dag4652 = OutputStructure_10_4652;
dag4661 = OutputStructure_10_4661;

dag3091 = OutputStructure_11_3091;
dag3092 = OutputStructure_11_3092;
dag3102 = OutputStructure_11_3102;
dag3104 = OutputStructure_11_3104;

dag6173 = OutputStructure_12_6173;
dag6204 = OutputStructure_12_6204;
dag6234 = OutputStructure_12_6234;
dag6279 = OutputStructure_12_6279;

dag125 = OutputStructure_13_125;
dag126 = OutputStructure_13_126;
dag136 = OutputStructure_13_136;
dag3229 = OutputStructure_13_3229;

dag4 = OutputStructure_14_4;
dag32 = OutputStructure_14_32;
dag36 = OutputStructure_14_36;
dag39 = OutputStructure_14_39;



%%

dlm_basic_eval(dlm,dag1336,1336);
dlm_basic_eval(dlm,dag1338,1338);
dlm_basic_eval(dlm,dag1340,1340);
dlm_basic_eval(dlm,dag1350,1350);

dlm_basic_eval(dlm,dag4774,4774);
dlm_basic_eval(dlm,dag4782,4782);
dlm_basic_eval(dlm,dag4793,4793);
dlm_basic_eval(dlm,dag4804,4804);

dlm_basic_eval(dlm,dag4647,4647);
dlm_basic_eval(dlm,dag4651,4651);
dlm_basic_eval(dlm,dag4652,4652);
dlm_basic_eval(dlm,dag4661,4661);

dlm_basic_eval(dlm,dag3091,3091);
dlm_basic_eval(dlm,dag3092,3092);
dlm_basic_eval(dlm,dag3102,3102);
dlm_basic_eval(dlm,dag3104,3104);

dlm_basic_eval(dlm,dag6173,6173);
dlm_basic_eval(dlm,dag6204,6204);
dlm_basic_eval(dlm,dag6234,6234);
dlm_basic_eval(dlm,dag6279,6279);

dlm_basic_eval(dlm,dag125,125);
dlm_basic_eval(dlm,dag126,126);
dlm_basic_eval(dlm,dag136,136);
dlm_basic_eval(dlm,dag3229,3229);

dlm_basic_eval(dlm,dag4,4);
dlm_basic_eval(dlm,dag32,32);
dlm_basic_eval(dlm,dag36,36);
dlm_basic_eval(dlm,dag39,39);


     
