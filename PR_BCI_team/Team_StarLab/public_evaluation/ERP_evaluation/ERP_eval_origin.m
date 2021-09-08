%%
ival_cfy_fixing = 1; % 1:ture // fixing
BTB.method = 'cap_epo'; %cap_epo_i_cICA_PCA_AF
eval(sprintf('EPO = %s;',BTB.method))
modal = BTB.method(1:3);

excel_AUC = [];
excel_SNR = [];
excel_SNR_db = [];
slt_ival = [];

for j=1:10
%% select channels
% erp_chan=15:28;
chan =  {'C3','C1','C2','C4','CP1','CP2','P3','Pz','P4','PO7','PO3','POz','PO4','PO8','O1','Oz','O2'};
% eval(sprintf('chan = %s_inter_chan{j};',modal));

%% ival setting

ref_ival= [-200 0] ;
r_ival = [100 600];
% psn_ival = [-150 -50] ;
psn_ival = [-200 0] ;
p300_ival = [280 380];

ival_cfy = [200 250; 250 300; 300 350; 350 400; 400 450];
% if j==1
%     ival_cfy = ival_cfy-500;
%     ref_ival= ref_ival-500 ;
%     r_ival = r_ival-500;
%     psn_ival = psn_ival-500 ;
%     p300_ival = p300_ival-500;
% end

%% training
eval(sprintf('epo = %s_epo{j,1};',modal));

epo = proc_selectChannels(epo, chan);

% select ival
if ival_cfy_fixing == false
epo_r= proc_selectIval(epo, r_ival, 'IvalPolicy','minimal');
epo_r= proc_rSquareSigned(epo_r);
ival_cfy= procutil_selectTimeIntervals(epo_r);
disp('ival cfy non-fixing  -  Check ival_cfy')
end

slt_ival{j} = ival_cfy;

fv_Tr{j,1}= proc_baseline(epo, ref_ival);
fv_Tr{j,1}= proc_jumpingMeans(fv_Tr{j,1}, ival_cfy);

xsz= size(fv_Tr{j,1}.x);
fvsz= [prod(xsz(1:end-1)) xsz(end)];
% classifier C train_RLDAshrink
C = train_RLDAshrink(reshape(fv_Tr{j,1}.x,fvsz), fv_Tr{j,1}.y);

for speedIdx = 3:sum(~cellfun('isempty', cap_cnt(j,:)))
%% test
epo = EPO{j,speedIdx};

epo = proc_selectChannels(epo, chan);
% epo = proc_selectChannels(epo, erp_chan);

fv_Te{j,speedIdx-2}= proc_baseline(epo, ref_ival);
fv_Te{j,speedIdx-2}= proc_jumpingMeans(fv_Te{j,speedIdx-2}, ival_cfy);

xTesz= size(fv_Te{j,speedIdx-2}.x);
% test loss
outTe= apply_separatingHyperplane(C, reshape(fv_Te{j,speedIdx-2}.x, [prod(xTesz(1:end-1)) xTesz(end)]));
lossTe = mean(loss_0_1(fv_Te{j,speedIdx-2}.y, outTe));
%         lossSem= std(lossTe,0,1)/sqrt(size(lossTe,1));

% training loss
outTr= apply_separatingHyperplane(C, reshape(fv_Tr{j,1}.x, fvsz));
% [loss_all,pred_results{j,speedIdx}, y_labels{j,speedIdx}] = loss_0_1(fv_Tr.y, outTr);
[loss_all] = loss_0_1(fv_Tr{j,1}.y, outTr);
lossTr = mean(loss_all);

% outTe= 0.5*sign(outTe)+1.5;

mc = calcConfusionMatrix(epo.y, outTe, lossTe);
TPR = mc(1,1)/(mc(1,1)+mc(2,1));
FPR = mc(2,2)/(mc(2,2)+mc(1,2));

% outTe= (outTe - 1.0);

[ERP_per.roc, ERP_per.auc]= roc_curve(epo.y, outTe,'plot',0);

excel_AUC(speedIdx-1, j) = ERP_per.auc;

[X,Y,T,AUC] = perfcurve(epo.y(2,:),outTe,0);
excel_AUC2(speedIdx-1, j) = AUC; 

end
end
excel_SNR_all = excel_SNR;
%%
disp('Mean AUC')
for ispeed = 1:4
mean_AUC(ispeed) = sum(excel_AUC(ispeed,:))/nnz(excel_AUC(ispeed,:));
end
disp(mean_AUC)

