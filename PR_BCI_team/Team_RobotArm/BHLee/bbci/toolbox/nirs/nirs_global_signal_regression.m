function data=nirs_global_signal_regression(data)
% nirs_global_signal_regression - regresses a global mean and slow sinusoidal frequencies 
% in order to reduce the global (e.g. extra-cerebal) artifacts
%
% Synopsis:
%   data=nirs_global_signal_regression(data)
%   data=nirs_global_signal_regression(data,hpfilter)
%
% Arguments:
%   data: data
%   hpfilter: cut-off time of high pass filter in seconds. default: 128 seconds
%
% Returns:
%   data structure with global mean and slow oscillations regressed out.
%
%note: data should be low passed filtered and reduced to only meaningful
%channels.
%
% jan.mehnert@charite.de 2011

signals = intersect(fieldnames(data),{'x' 'oxy' 'deoxy'});

data.GSR = struct();

%setting time for high-pass filter
if ~exist('hpfilter','var')
    hpfilter=128;
else
    hpfilter=data.FLThp;
end

for ii=1:numel(signals)

sig = signals{ii};

%calculate the global mean
GM=mean(zscore(detrend(data.(sig))),2);

%add offset
desmat=ones(length(GM),1); %OFFSET

%create cosines used as high-pass filter
t=(1:size(desmat,1))*1/data.fs; %SLOW-FILTERS
tmax=max(t);
Nfilters=floor((1.8/hpfilter)/(1/tmax));
filters=  cos( ((  repmat(t(:),1, Nfilters ) .* ( repmat([1:Nfilters],length(t),1)) ) / (tmax) ).*pi ) ;

%create design matrix
desmat=[desmat filters GM];


%regression
beta        =desmat\data.(sig); %fitting
dmean=mean(data.(sig),1);%mean

for i=1:size(data.(sig),2) %subtraction
    data.(sig)(:,i)=data.(sig)(:,i)-desmat*beta(:,i) + dmean(i) ;
end

data.GSR(ii).design_matrix = desmat;
data.GSR(ii).beta=beta;
data.GSR(ii).signal = sig;

end