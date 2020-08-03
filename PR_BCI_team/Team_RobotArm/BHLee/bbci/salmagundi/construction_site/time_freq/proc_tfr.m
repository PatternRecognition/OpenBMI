function [epotfr, freqline]= proc_tfr(epo, varargin)

if ~exist('tfrstft', 'file'),
  addpath('~/matlab/Import/timeFreq');
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'tf_fcn', 'tfrstft', ...
                  'fs', 50, ...
                  'hz_resol', 1, ...
                  'hz_range', [5 ceil(epo.fs/3)]);

[T, nChans, nEvents]= size(epo.x);
freqresol= epo.fs/T;
nSubFreq= round(opt.hz_resol/freqresol);
actualresol= freqresol*nSubFreq;
freqline= 0:actualresol:epo.fs/2;
freqidx= find(freqline>=opt.hz_range(1) & freqline<= ...
              opt.hz_range(2));
freqline= freqline(freqidx);
nSubTime= round(epo.fs/opt.fs);
T1= floor(T/nSubTime);
epotfr= copy_struct(epo, 'not','x','t','xUnit','yUnit','fs');

tic;
for ce= 1:nChans*nEvents,
  [tfr]= abs(feval(opt.tf_fcn, epo.x(:,ce)));
  tfr= tfr(1:ceil(T/2),:);          %% throw away neg frequencies
  jm= jumpingMean(tfr, nSubFreq);
  jm= jumpingMean(jm', nSubTime);
  jm= jm(:,freqidx);
  if ce==1,
    nFreqs= size(jm, 2);  
    epotfr.x= zeros([T1*nFreqs nChans nEvents]);
  end
  jm= reshape(jm, [prod(size(jm)) 1]);
  epotfr.x(:,ce)= jm;
  print_progress(ce, nChans*nEvents);
end

epotfr.x= reshape(epotfr.x, [T1*nFreqs nChans nEvents]);
epotfr.dim= [T1 nFreqs];
epotfr.t= {jumpingMean(epo.t(:),nSubTime)', freqline};
epotfr.xUnit= {'ms','Hz'};
epotfr.yUnit= 'au';
epotfr.fs= epo.fs/nSubTime;
