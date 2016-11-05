function dat= func_powerspectrum (data, band, varargin)
%PROC_SPECTRUM -  calculate the power spectrum
%
%dat= proc_spectrum(dat, band, <win/N, step>)
%dat= proc_spectrum(dat, band, <opts>)
% IN   dat  - data structure of continuous or epoched data
%      band - frequency band
%      win  - window for FFT
%      N    - window width for FFT -> square window, default dat.fs
%      step - step for window (= # of overlapping samples), default N/2
%      opt  - struct of options:
%       .win       - window for FFT, default ones(dat.fs, 1)
%       .step      - step for window, default N/2
%       .db_scaled - boolean, if true values are db scaled (10*log10),
%                    default true

%%
% data
dat=data;
ban=band;
opt=opt_cellToStruct(varargin{:});
epo=struct('win',[],'N',[],'step',[],'scale',[]);

if isempty(dat)
    warning('Warning! data is empty');
end
if isempty(ban)
    warning('Band is not exist.');
end

if ~isfield(opt,'win') 
   epo.win=dat.fs;
else
    epo.win=opt.win;
end

if ~isfield(opt,'N')
   epo.N=dat.fs;
else
    epo.N=opt.N;
end

if ~isfield(opt,'step')
   epo.step=dat.fs/2;
else
   epo.N=opt.N;
end

if ~isfield(opt,'scale')
   epo.scale='db';
else
    epo.scale=opt.scale;
end

[T, nEvents , nChans]= size(dat.x);

if length(epo.win)==1
    if epo.win>T
        warning('window legth is higher than signal')
    end
    epo.win=ones(epo.win,1);
end
N=length(epo.win);
normwin=norm(epo.win);
Freq=(0:N)/2*dat.fs/N;

%%
XX= zeros(N, nChans*nEvents);
nWindows= 1 + max(0, floor((T-N)/epo.step));
iv= 1:min(N, T);
Win= repmat(epo.win(:), [1 nChans*nEvents]);
bInd= band(1): ban(2);

%%calculate file

switch(lower(epo.scale)),
    case 'db',
        for iw= 1:nWindows,
            XX= XX + abs(fft(dat.x(iv,:).*Win, N)).^2;
            iv= iv + epo.step;
        end
        XX = XX/(nWindows*normwin^2);
        dat.x= reshape( 10*log10( XX(bInd,:)+eps ), [length(bInd), nChans, nEvents]);
        dat.yUnit= 'dB';
    case 'power',
        for iw= 1:nWindows,
            XX= XX + abs(fft(dat.x(iv,:).*Win, N).^2);
            iv= iv + epo.step;
        end
        dat.x= reshape(XX(bInd,:)/(nWindows*normwin^2), [length(bInd), nChans, nEvents]);
        dat.yUnit= 'power';
end



end

