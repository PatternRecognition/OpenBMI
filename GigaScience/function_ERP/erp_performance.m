function [ final_acc, final_itr ] = erp_performance(CNT, params)
opt = opt_cellToStruct(params);
%%
rc_order = importdata('random_cell_order.mat');
cpr = opt.Nsequence * size(rc_order{1,1}, 1);

%%
cnt_off= prep_selectChannels(CNT{1}, {'Index',opt.channel_index});
cnt_off_filt=prep_filter(cnt_off, {'frequency', opt.band});
smt_off=prep_segmentation(cnt_off_filt, {'interval', opt.segTime});
smt_off=prep_baseline(smt_off, {'Time',opt.baseTime});
smt_off_select=prep_selectTime(smt_off, {'Time',opt.selTime});
%%
fv_off=func_featureExtraction(smt_off_select,{'feature','erpmean';'nMeans',opt.nFeature});
[nDat, nTrials, nChans]= size(fv_off.x);
fv_off.x= reshape(permute(fv_off.x,[1 3 2]), [nDat*nChans nTrials]);
[clf_param] = func_train(fv_off,{'classifier','LDA'});
clear cnt_off cnt_off_filt smt_off smt_off_select fv_off nDat nTrials nChans
%%
cnt=prep_selectChannels(CNT{2},{'Index',opt.channel_index});
cnt =prep_filter(cnt, {'frequency', opt.band});
smt=prep_segmentation(cnt, {'interval', opt.segTime});
smt=prep_baseline(smt, {'Time',opt.baseTime});
smt=prep_selectTime(smt, {'Time',opt.selTime});
%%
dat.x= smt.x;
dat.fs = smt.fs;
dat.ival = smt.ival;
dat.t = smt.t;

for add=1:cpr:length(dat.t)
    dat_char_all.x(:,:,:,opt.init_speller_length)=dat.x(:,[add:add+cpr-1],:);
    dat_char_all.t(:,:,:,opt.init_speller_length)=dat.t(:,[add:add+cpr-1]);
    opt.init_speller_length=opt.init_speller_length+1;
end

for char = 1:length(opt.spellerText_on)
    in_nc=1;
    nc=1;
    nSeq=1;
    DAT=cell(1,length(opt.speller_text));
    tm_Dat=zeros(length(opt.speller_text),size(clf_param.cf_param.w,1));
    
    dat_char.x = dat_char_all.x(:,:,:,char);
    dat_char.t = dat_char_all.t(:,:,:,char);
    dat_char.fs = dat.fs;
    dat_char.ival = dat.ival;
    
    ft_dat=func_featureExtraction(dat_char,{'feature','erpmean';'nMeans',opt.nFeature});
    [nDat, nTrials, nCh]= size(ft_dat.x);
    ft_dat.x = reshape(permute(ft_dat.x,[1 3 2]), [nDat*nCh nTrials]);
    
    for i=1:nTrials
        for i2=1:size(rc_order{nSeq},2)
            DAT{rc_order{nSeq}(in_nc,i2)}(end+1,:) = ft_dat.x(:,nc);
        end
        for i2=1:length(opt.speller_text)
            if size(DAT{i2},1)==1
                tm_Dat(i2,:)=DAT{i2};
            else
                tm_Dat(i2,:)=mean(DAT{i2});
            end
        end
        
        CF_PARAM = clf_param;
        [Y]=func_predict(tm_Dat', CF_PARAM);
        [a1 a2]=min(Y);
        t_char(char,nSeq)= opt.speller_text{a2};
        
        nc=nc+1;
        in_nc=in_nc+1;
        if in_nc>size(rc_order{nSeq},1)
            in_nc=1;
            nSeq=nSeq+1;
        end
    end
    clear DAT tm_Dat Y a b a1 a2
end
clear add CF_PARAM char clf_param cnt dat dat_char dat_char_all tm_Dat Y DAT
clear ft_dat fv i i2 in_nc ival nc nCh nDat nSeq nTrials smt temp3 nChans
clear BMI EEG field file file3 marker

for seq=1:opt.Nsequence
    for nchar=1:length(opt.spellerText_on)
        seq_acc(nchar, seq)=strcmp(opt.spellerText_on(nchar),t_char(nchar,seq));
    end
end

sub_acc=sum(seq_acc)/length(opt.spellerText_on);
allsub_acc=sub_acc;

for i=1:opt.Nsequence
    t2(i)=opt.one_seq_time*i;
end

speller_number = length(opt.spellerText_on);
no_b = sub_acc.*log2(sub_acc) + (1-sub_acc).*log2((1-sub_acc)/(speller_number-1)) + log2(speller_number);
ind = find(isnan(no_b) == 1);
no_b(ind) = log2(speller_number);
no_itr = [no_b ./ (t2./60)];
allsub_ITR=no_itr;

init_speller_length=1;
clear seq_acc i ind n nchar sub_acc no_b no_itr seq t2 t_char2 t_char22 speller_length  cnt_off cnt_off_filt smt_off smt_off_select

%%
final_acc = allsub_acc(5)';
final_itr = allsub_ITR(5)';
end

