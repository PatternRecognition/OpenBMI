function cnt= proc_betClassifierOutput(cnt)
%PROC_BETCLASSIFIEROUTPUT - Add classifier output from bbci_bet experiment
%
%Synopsis:
% CNT= proc_betClassifierOutput(CNT)

[ctrllog]= eegfile_loadMatlab(cnt.file, 'vars','log');

ctrl= strukt('fs',cnt.fs, ...
             'clab',{'out'}, ...
             'x',NaN*zeros([size(cnt.x,1) 1]));
xi= ctrllog.cls.pos;
xi_start= xi(1);
xi= xi-xi_start+1;

lag= min(diff(ctrllog.cls.pos));
for k= 0:lag-1,
  ctrl.x(xi+k,1)= ctrllog.cls.values;
end

%% due to reduces sampling freq (fs=25Hz in ctrllog) might need to prune cnt
nSamples= size(cnt.x,1);
ctrl.x(nSamples+1:end,:)= [];

cnt= proc_appendChannels(cnt, ctrl);
