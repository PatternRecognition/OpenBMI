eeg_file= 'VPiac_10_10_13/CenterSpellerMVEP_VPiac';
[cnt, mrk_orig]= eegfile_loadMatlab(eeg_file, 'vars',{'cnt','mrk_orig'});

clab= {'F3','Fz','F4', 'C3','Cz','C4', 'P3','Pz','P4'};
cfy_ival= [90 110; 110 150; 150 250; 250 400; 400 750];
% Generate random classifier of correct format
C= struct('b',0);
C.w= randn(length(clab)*size(cfy_ival,1), 1);

bbci= struct;
bbci.source.acquire_fcn= @bbci_acquire_offline;
bbci.source.acquire_param= {cnt, mrk_orig};

bbci.signal.clab= clab;

bbci.feature.proc= {{@proc_baseline, [-200 0]}, ...
                    {@proc_jumpingMeans, cfy_ival}};
bbci.feature.ival= [-200 750];

bbci.classifier.C= C;

bbci.control.fcn= @bbci_control_ERP_Speller;
bbci.control.param= {struct('nClasses',6, 'nSequences',10)};
bbci.control.condition.marker= [11:16,21:26,31:36,41:46];

bbci.log.output= 0;

bbci.quit_condition.running_time= 60;

for k= [2 5 10 25 50 75 100 150 200:100:1000],
  bbci.signal.buffer_size= k*1000;
  tic;
  bbci_apply(bbci);
  fprintf('%4d -> %.2f\n', k, toc);
end


%   2 -> 1.92
%   5 -> 1.92
%  10 -> 1.93
%  25 -> 1.97
%  50 -> 2.40
%  75 -> 3.00
% 100 -> 3.57
% 150 -> 6.21
% 200 -> 7.71
% 300 -> 10.45
% 400 -> 13.20
% 500 -> 15.85
% 600 -> 18.68
% 700 -> 21.12
% 800 -> 23.97
% 900 -> 26.64
%1000 -> 29.32
