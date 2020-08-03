EXPORT_DIR= '/home/nibbler/blanker/Daten/eegExport/';

load([EXPORT_DIR 'selfpaced2s_aa01_100Hz_sol']);

clear epo
epo.x= y;
epo. fs= fs;
epo.clab= {'F3','F1','Fz','F2','F4', ...
           'FC5','FC3','FC1','FCz','FC2','FC4','FC6', ...
           'C5','C3','C1','Cz','C2','C4','C6', ...
           'CP5','CP3','CP1','CPz','CP2','CP4','CP6', 'O1'};
epo.y= [z==-1; z==1];

test_idx= find(z==0);
train_idx= find(ismember(z, [-1 1]));

pTape= 'p_nips01_scp_many';
mTape= 'c_nips01_lin';

[pn, classy, epMean, epStd, emMean, emStd]= ...
    selectProcessFromTape(epo, pTape, mTape, [5 10], 1);
eval(getBlockFromTape(pTape, pn));

C= trainClassifier(fv, classy);
out= applyClassifier(fv, classy, C);
train_err= 100*mean(sign(out(train_idx))~=z(train_idx))
test_err= 100*mean(sign(out(test_idx))~=zz(test_idx))


epo.y= [zz==-1; zz==1];
doXvalidation(fv, classy, [10 10]);
