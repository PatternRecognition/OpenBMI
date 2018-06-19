function [ Acc ] = ssvep_performance(CNT , params)
opt = opt_cellToStruct(params);
%% CCA - Anaysis
SMT=[];
for onoff=1:2
    cnt = CNT{onoff};
    cnt=prep_filter(cnt, {'frequency', opt.band});    
    CNTch = prep_selectChannels(cnt, {'Index', opt.channel_index});
    SMT_iter = prep_segmentation(CNTch, {'interval', opt.time_interval});
    if onoff==1
        SMT= SMT_iter;
        clear SMT_iter
    else
        SMT = prep_addTrials(SMT, SMT_iter);
        clear SMT_iter
    end
end

tot = size(SMT.x, 2);
count1= tot;
for i = 1: size(SMT.x, 2)
    res_cca = ssvep_cca_analysis(squeeze(SMT.x(:,i,:)),{'marker',opt.marker;'freq', opt.freq;'fs', opt.fs;'time',opt.time});
    [~, ind] = max(res_cca);
    if SMT.y_dec(i) ~= ind
        count1 = count1 -1;
    end
end
Acc =count1/tot;
clear CNTch SMT tot count1 i res_cca in
end
