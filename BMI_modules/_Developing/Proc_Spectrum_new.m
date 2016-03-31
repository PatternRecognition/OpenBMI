function dat= Proc_Spectrum_new (data, band, varargin)
%PROC_SPECTRUM -  calculate the power spectrum
%
%dat= proc_spectrum(dat, band, <win/N, step>)
%dat= proc_spectrum(dat, band, <opts>)
%
%
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
%
% OUT  dat  - updated data structure
dat = data;
ban = band;
opt = opt_cellToStruct(varargin{:});

[T, nEvents , nChans]= size(dat.x);

if ~isfeield(dat)
    warning('Warning! data is empty');
end
if ~isfeield(ban)
    warning('


% props= {'Win'     dat.fs   'DOUBLE'
%         'Step'    []       'INT'
%         'Scaling' 'db'     'CHAR(db power normalized unnormalized complex)'
%        };
% [opt, isdefault]= opt_setDefaults(opt, props, 1);


Win=ones(dat.fs); %%기본 윈도우 일때
% Win=ones(varagin에서 들어올때)
[T, nEvents , nChans]= size(dat.x);

%% 추가해야됨
A

N= length(Win);
spectrum


fs=dat.fs;
% dat.fs
normWin=norm(N);
N=100;
step=N/2;


nB=size(band);
Freq=(0:N/2)*fs/N;
bInd=[2: 41];
Sf=band(1);
Ef=band(2);

Freq=Freq(Sf+1:Ef+1);

XX= zeros(N, nChans*nEvents);

iv= 1:min(N, T);
Win= repmat(Win(:), [1 nChans*nEvents]);
nWindows= 1 + max(0, floor((T-N)/step));

for iw= 1:nWindows,
            XX= XX + abs(fft(dat.x(iv,:).*Win, N)).^2;
            iv= iv + step;
        end
        XX = XX/(nWindows*normWin^2);
        dat.x= reshape( 10*log10( XX(bInd,:)+eps ), [length(bInd), nChans, nEvents]);
        dat.yUnit= 'dB';

dat.t= Freq;
dat.xUnit= 'Hz';

end

