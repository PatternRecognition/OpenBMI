file_list= {'Klaus_04_04_08/imag_basketfbKlaus', 
            {'Klaus_04_03_11/imag_basketfb1Klaus',
             'Klaus_04_03_11/imag_basketfb2Klaus'},
            'Guido_04_03_29/imag_1drfbGuido'};

szenario_list= {'imag_lettKlaus_udp_1d_szenario_01',
                'imagKlaus_udp_brainbasket_szenario_02',
                'imag_lettGuido_udp_1d_szenario_01'};

export_list= {'basket_3_targets', 
              'basket_6_targets',
              '1d_relative'};

for ff= 1:length(file_list),
file= file_list{ff};
szenario= szenario_list{ff};

if iscell(file),
  [cnt, mrk, mnt]= loadProcessedEEG(file{1});
  for aa= 2:length(file),
    [cnt2, mrk2]= loadProcessedEEG(file{aa});
    mrk2.info(2,:)= mrk2.info(2,:) + size(cnt.x,1);
    mrk_info= cat(2, mrk.info, mrk2.info);
    [cnt, mrk]= proc_appendCnt(cnt, cnt2, mrk, mrk2);
    mrk.info= mrk_info;
  end
else  
  [cnt, mrk, mnt]= loadProcessedEEG(file);
end
mrk= mrk_selectClasses(mrk, 'not','hit','miss');

override= {};
if ~isempty(strmatch('Guido_04_03_29', cnt.title)),
  override= {'dscr.C.setting= 0'};
end

if ~isempty(strmatch('Klaus_04_03_11/imag_basket', cnt.title)),
%% what happened here?
  corrupt= find(mrk.info(1,:)==14);
  mrk.info(2,corrupt)= 32730;
end

%% for both Klaus experiments there should be bias -2 added to
%% the classifier output
cnt= proc_addClassifierOutput(cnt, szenario, override{:});
cnt= proc_addChannelFromLog(cnt, mrk);

saveProcessedEEG([EEG_EXPORT_DIR 'for_florin_' export_list{ff}], ...
                 cnt, mrk, mnt);

end


return


%% basket data with three targets:
%%  time_before_free 520, trial_duration 2000, (time_after_hit 120)
%% basket data with six targets:
%%  time_before_free 520, trial_duration 2000 [last run 2500], 
%%  (time_after_hit 160, before_next 520)

fb= proc_selectChannels(cnt, 'out','yData');
epo= makeEpochs(fb, mrk, [0 2520]);
valid= find(any(~isnan(epo.x(:,2,:)),1));
epo= proc_selectEpochs(epo, valid)
erp= proc_average(epo);

subplot(1,2,1);
plot(erp.t, squeeze(erp.x(:,1,:)));
legend(erp.className, 2);
title('classifier output');

subplot(1,2,2);
plot(erp.t, squeeze(erp.x(:,2,:)));
legend(erp.className, 2);
title('basket feedback trajectory');



%% 1d relative
%%  caveat: trials have different durations

fb= proc_selectChannels(cnt, 'out','xData');
mk= mrk_selectClasses(mrk, 'left','right','foot');
epo= makeEpochs(fb, mk, [0 1500]);
valid= find(any(~isnan(epo.x(:,2,:)),1));
epo= proc_selectEpochs(epo, valid)
erp= proc_average(epo);

subplot(1,2,1);
plot(erp.t, squeeze(erp.x(:,1,:)));
legend(erp.className, 2);
title('classifier output');

subplot(1,2,2);
plot(erp.t, squeeze(erp.x(:,2,:)));
legend(erp.className, 2);
title('1d rel feedback trajectory');


return


%% basic script for plotting average trajectories

%% make epochs of 2.5 seconds
iv= 0:2500*cnt.fs/1000;
[nClasses, nEvents]= size(mrk.y);
T= length(iv);
IV= iv(:)*ones(1,nEvents) + ones(T,1)*mrk.pos;

%% classifier output is the last but 2 channel
out= reshape(cnt.x(IV,end-2), size(IV));

out_avg= zeros(T, nClasses);
for cc= 1:nClasses,
  idx= find(mrk.y(cc,:));
  out_avg(:,cc)= mean(out(:,idx),2);
end

plot(out_avg);


