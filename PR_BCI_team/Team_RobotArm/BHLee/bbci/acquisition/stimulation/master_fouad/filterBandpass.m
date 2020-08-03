% chan:   'C4'
%         {'C4','Pz'}
%         'C#'
%         'C*'
%         'C3,z,4'
%         'C3,z,4','P3,z,4'
%         'C3-4'

function [trial_spec t_trial data_ct_filt trial mean_trial data_set_meantrial data_set_meantrialSpec] = filterBandpass(cnt,mrk,mrk_num,chan,Para)

mrk_pos = get_mrkPos(mrk,mrk_num);

length_trial = Para.ref_time+(Para.act_time+Para.ifi)*length(Para.modfreq);
t_trial = [0:1/cnt.fs:length_trial-(1/cnt.fs)];

ct= proc_selectChannels(cnt, chan);
num_chan = size(ct.x,2);
trial = cell(2,num_chan);
trial_mean = cell(2,num_chan);
for ii = 1:length(Para.modfreq)
  
    low_freq = (Para.modfreq(ii)-1);
    high_freq = (Para.modfreq(ii)+1);
    [b,a]= butter(5, [low_freq high_freq]/cnt.fs*2);
    ct_filt = proc_filtfilt(ct,b,a);
    
    data_ct_filt = ct_filt.x;
    data_clab_filt = ct_filt.clab;
    for i = 1:num_chan
      trial{1,i} = data_clab_filt(i);
      trial{2,i} = cntToEpoch(data_ct_filt(:,i),mrk_pos.S102,length_trial,ct_filt.fs);
     
      mean_trial{1,i} =  data_clab_filt(i);
      mean_trial{2,i} = mean(trial{2,i},2);
      

      
      trial_spec{1,i} =  data_clab_filt(i);
      trial_spec{2,i} = abs(fft(trial{2,i}));
      
      mean_trial_spec{1,i} =  data_clab_filt(i);
      mean_trial_spec{2,i} = mean(trial_spec{2,i},2);
      %       mean_trial(:,i) = mean(trial,2);
%     
%       trial_cell{1,ii} = Para.modfreq(ii);
%       trial_cell{2,ii} = trial;

    end
    data_set_meantrial{1,ii} = Para.modfreq(ii);
    data_set_meantrial{2,ii} = mean_trial;
    
    data_set_meantrialSpec{1,ii} = Para.modfreq(ii);
    data_set_meantrialSpec{2,ii} = mean_trial_spec;
    
end