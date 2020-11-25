function total_params = dlm_basic_eval(dlm,dag,FileNum)
% Zach A. Pardos (zp@csail.mit.edu)
% Bayesian Prediction Analysis for Dynamic Learning Maps (Kansas)
% in collaboration with Neil Heffernan and colleagues  (WPI)
% data property of Angela Broaddus (Kansas) and Neal Kingston (Kansas)

dlm.responses = dlm.responses(dlm.complete,:);
dlm.studentfold5 = dlm.studentfold5(dlm.complete);

N=size(dag,1);
hidden_nodes = 1:N-size(dlm.responses,2);
observed_nodes = N-size(dlm.responses,2)+1:N;

eclass=1:N;
bnet = mk_bnet(dag,2*ones(1,N),'observed',observed_nodes,'discrete',1:N,'equiv_class',eclass ); 

total_params=0;
for C=1:N
 bnet.CPD{C} = tabular_CPD(bnet, C);
 cpt = CPD_to_CPT(bnet.CPD{C});
 total_params = total_params + length(cpt(:));
end
total_params =total_params/2;


cases = cell(size(dlm.responses,1),N);
cases(:,observed_nodes) = num2cell(dlm.responses+1);

engine = jtree_inf_engine(bnet);
max_iter = 300;
filename = ['dlm_resultsFinalRun',num2str(FileNum),'.txt'];
report = fopen(filename,'w');
for fold=1:5
    cases2=cases(dlm.studentfold5 ~= fold,:);
    [bnet2, LLtrace, engine2] = learn_params_em(engine,cases2',max_iter);

    cases3=cases(dlm.studentfold5 == fold,:);
    for c=1:size(cases3,1)
	fprintf('%d%% done with fold %d of 5\n',round(c*100/size(cases3,1)),fold);
        case1=cases3(c,:);
        for ifold=1:3
            scase=cell(1,N);
            evitems=dlm.itemfold3 ~= ifold;
            scase(observed_nodes(evitems)) = case1(observed_nodes(evitems));
            [engine3,ll] = enter_evidence(engine2,scase);
            titems=find(dlm.itemfold3 == ifold);
            for t=titems'
                m = marginal_nodes(engine3,observed_nodes(t));
                m = m.T(2);
                fprintf(report,'%d %d %d %.5f %d\n',fold,c,t,m,case1{observed_nodes(t)}-1);
            end
        end
    end
end

fclose(report);