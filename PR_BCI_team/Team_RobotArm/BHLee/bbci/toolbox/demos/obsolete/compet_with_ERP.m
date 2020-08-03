EXPORT_DIR= '/home/nibbler/blanker/Daten/eegExport/';
load([EXPORT_DIR 'selfpaced2s_aa01_100Hz_sol']);

epo.x= y;
epo.y= zeros(2, length(z));
epo.y(1,z==-1)= 1;
epo.y(2,z==1)= 1;
epo.t= 10*round(1000*x/10);
epo.clab= {'F3','F1','Fz','F2','F4', ...
           'FC5','FC3','FC1','FCz','FC2','FC4','FC6', ...
           'C5','C3','C1','Cz','C2','C4','C6', ...
           'CP5','CP3','CP1','CPz','CP2','CP4','CP6','O1'};
epo.fs= 100;

epo= proc_baseline(epo, [-1000 -800]);

test_idx= find(z==0);
train_idx= find(ismember(z, [-1 1]));

fv= proc_laplace(epo, 'small', '', 'CP#');
fv= proc_bipolarChannels(fv, {'C3-C2','CP3-CP2'});
fv= proc_selectIval(fv, [-200 -120]);
fv= proc_flaten(fv);

C= train_gaussianERPmodel(fv.x, fv.y);
out= apply_separatingHyperplane(C, fv.x);
train_err= 100*mean(sign(out(train_idx))~=z(train_idx))
test_err= 100*mean(sign(out(test_idx))~=zz(test_idx))
