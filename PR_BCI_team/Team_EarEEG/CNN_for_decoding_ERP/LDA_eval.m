%%
ival_cfy_fixing = 1; % 1:ture // fixing

excel_AUC = [];
excel_SNR = [];
excel_SNR_db = [];
slt_ival = [];

for subNum=1:15
%% setting
% select channels
chan = {'C3','C1','C2','C4','CP1','CP2','P3','Pz','P4','PO7','PO3','POz','PO4','PO8','O1','Oz','O2'};

% ival setting

ref_ival= [-200 0] ;
r_ival = [100 600];
% psn_ival = [-150 -50] ;
psn_ival = [-200 0] ;
p300_ival = [280 380];

% ival_cfy = [200 250; 250 300; 300 350; 350 400; 400 450];
ival_cfy = [200 225; 225 250; 250 275; 275 300; 300 325; 325 350; 350 375; 375 400; 400 425; 425 450];


%% training
epo = epo_train{subNum};
epo = proc_selectChannels(epo, chan);

% select ival
if ival_cfy_fixing == false
epo_r= proc_selectIval(epo, r_ival, 'IvalPolicy','minimal');
epo_r= proc_rSquareSigned(epo_r);
ival_cfy= procutil_selectTimeIntervals(epo_r);
disp('ival cfy non-fixing  -  Check ival_cfy')
end

slt_ival{subNum} = ival_cfy;

fv_Tr= proc_baseline(epo, ref_ival);
fv_Tr= proc_jumpingMeans(fv_Tr, ival_cfy);

xsz= size(fv_Tr.x);
fvsz= [prod(xsz(1:end-1)) xsz(end)];

% classifier C train_RLDAshrink
C = train_RLDAshrink(reshape(fv_Tr.x,fvsz), fv_Tr.y);

%% test
epo = epo_test{subNum};

epo = proc_selectChannels(epo, chan);

fv_Te= proc_baseline(epo, ref_ival);
fv_Te= proc_jumpingMeans(fv_Te, ival_cfy);

xTesz= size(fv_Te.x);
% test loss
outTe= apply_separatingHyperplane(C, reshape(fv_Te.x, [prod(xTesz(1:end-1)) xTesz(end)]));
lossTe = mean(loss_0_1(fv_Te.y, outTe));

% training loss
outTr= apply_separatingHyperplane(C, reshape(fv_Tr.x, fvsz));
lossTr = mean(loss_0_1(fv_Tr.y, outTr));

pred_prop(:,subNum) = outTe;
epo.y_dec = epo.y(1,:);
[ERP_per.roc, ERP_per.auc]= roc_curve(epo.y, outTe,'plot',0);

excel_AUC(subNum) = ERP_per.auc;

end
%%
disp('Mean AUC')
mean_AUC = sum(excel_AUC)/nnz(excel_AUC);

disp(mean_AUC)
