function dag = structure_14
% auto-generated code by Doug based on Zachs code
% Bayesian Prediction Analysis for Dynamic Learning Maps (Kansas)
% in collaboration with Neil Heffernan and colleagues  (WPI)
% data property of Angela Broaddus (Kansas) and Neal Kingston (Kansas)
N=39;

M1104=1;
M1289=2;
M1133=3;
M1135=4;
M1140=5;
M1118=6;
M1120_M1108=7;
M1122=8;
M1124=9;
M1105=10;
M1126=11;
M1106=12;
M1128=13;
M1127=14;
I1=15;
I2=16;
I3=17;
I4=18;
I5=19;
I6=20;
I7=21;
I8=22;
I9=23;
I10=24;
I11=25;
I12=26;
I13=27;
I14=28;
I15=29;
I16=30;
I17=31;
I18=32;
I19=33;
I20=34;
I21=35;
I22=36;
I23=37;
I24=38;
I25=39;

dag=zeros(N,N); 
dag(M1104, [M1289 I1 I2 I5 I6]) = 1;
dag(M1289, [M1133 M1118 I3 I4]) = 1;
dag(M1133, [M1135 M1140 I7 I8]) = 1;
dag(M1135, [M1118 M1105 I9]) = 1;
dag(M1140, [M1118 M1105 I10]) = 1;
dag(M1118, [M1120_M1108 M1122 I11]) = 1;
dag(M1120_M1108, [M1124 M1105 I12 I13 I15]) = 1;
dag(M1122, [M1124 I14]) = 1;
dag(M1124, I16) = 1;
dag(M1105, [M1106 M1128 I17 I18]) = 1;
dag(M1126, [M1128 I19]) = 1;
dag(M1106, [M1128 I20 I21]) = 1;
dag(M1128, [M1127 I22 I23]) = 1;
dag(M1127, [I24 I25]) = 1;
