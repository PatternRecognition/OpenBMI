function dat= func_spectrogram(data, frequnecy, varargin)
% func_spectrogram :
%  Calculating the spectrogram value using a Short-Time Fourier Transform
%    
% Example:
% dat = func_spectrogram(dat,frquency, {'win',100})
% dat = func_spectrogram(dat,frquency, {'noverlap',50})
% dat = func_spectrogram(dat,frquency, {'scale',{'db'})
% Input:
%      dat - Data structure of continuous or epoched data
%      band - Frequency band
%      
% Options:
%      win - Window for FFT
%      noverlap - Step for window, default N/2
%      scale - Set the scales of FFT (amlitude, power, db, phase) 
%
% Returns:
%     dat - Result of spectrogram value

%%
% data
dat=data;
opt=opt_cellToStruct(varargin{:});
epo=struct('win',[],'noverlap',[],'scale',[]);

if isempty(dat)
    error('OpenBMI: data is empty');
end
if isempty(frequnecy)
    warning('OpenBMI: frequency is not exist.');
end

if ~isfield(opt,'win') 
   epo.win=dat.fs/2;
else
    epo.win=opt.win;
end

if ~isfield(opt,'noverlap')
    epo.noverlap=(dat.fs)/2-1;
else
    epo.noverlap=opt.noverlap;
end


if isfield(opt,'scale')
   epo.scale='db';
else
    epo.scale=opt.scale;
end

[T, nEvents , nChans]= size(dat.x);

if length(epo.win)==1
    if epo.win>T
        warning('window legth is higher than signal')
    end
end
sz =size(dat.x);
X=dat.x;
dat.x=[];

%%
for chan=1:nChans
    [S,F,T] = spectrogram(X(:,nChans,:),epo.win,epo.noverlap,frequency,dat.fs);
    dat.x(:,:,chan)=S;
end

switch(lower(epo.scale)),
    case 'amplitude'
        dat.x = abs(dat.x);
        dat.yUnit= 'amplitude';
    case 'power'
        dat.x = abs(dat.x).^2;
        dat.yUnit= 'power';
    case 'db'
        dat.x = 10* log10( abs(dat.x).^2 );
        dat.yUnit= 'log power';
    case 'phase'
        dat.x = angle(dat.x);
        dat.yUnit= 'phase'; 
end


end
