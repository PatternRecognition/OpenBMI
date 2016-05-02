function dat= func_spectrogram(data, frequnecy, varargin)
% prep_powerspectrum : calculating the power spectrum in selected band

% IN   dat  - data structure of continuous or epoched data
%      band - frequency band
%      win  - window for FFT
%      N    - window width for FFT -> square window, default dat.fs
%      step - step for window (= # of overlapping samples), default N/2
%      opt  - struct of options:
%       .win       - window for FFT, default ones(dat.fs, 1)
%       .noverlap      - step for window, default N/2
%       .clab   
%       .scale - 

%%
% data
dat=data;
fre=frequnecy;
opt=opt_cellToStruct(varargin{:});
epo=struct('win',[],'noverlap',[],'clab',[],'scale',[]);

if isempty(dat)
    warning('[OpenBMI] Warning! data is empty');
end
if isempty(fre)
    warning('[OpenBMI] frequency is not exist.');
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
    [S,F,T] = spectrogram(X(:,nChans,:),epo.win,epo.noverlap,fre,dat.fs);
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
