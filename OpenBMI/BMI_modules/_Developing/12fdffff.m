function dat= Proc_Spectrum_new (data, band, varargin)
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
%
% OUT  dat  - updated data structure
dat = data;
ban = band;
opt={'win',dat.fs; 'N',dat.fs; 'step',dat.fs/2 ; 'scale','db'};
opt=opt_cellToStruct(opt)
[T, nEvents , nChans]= size(dat.x);

if isempty(dat)
    warning('Warning! data is empty');
end
if isempty(ban)
    warning('Band is not exist.');
end

if opt.win>T
    warning('Window size is longer than size');
end
opt.win=ones(dat.fs,1)
N= length(opt.win);
normWin  = norm(opt.win) ;
nWindows= 1 + max(0, floor((T-N)/opt.step));

%% X축 frequency 만들어주기
sb= size(band);
if isequal(sb, [2 1]), band= band'; end
% if sb(1)>1,
%     bInd= [];
%     for ib= 1:sb(1);
%         [bi, Freq]= procutil_getBandIndices(band(ib,:), dat.fs, N);
%         bInd= cat(2, bInd, bi);
%     end
%     return
% end

Freq= (0:N/2)*dat.fs/N;

if isempty(band)
    bInd= 1:N/2+1;
else
    ibeg= max(find(Freq<=band(1)));
    iend= min([find(Freq>=band(2)) length(Freq)]);
    bInd= ibeg:iend;
end

Freq= Freq(bInd);

%%
iv= 1:min(N, T);
Win= repmat(opt.win(:), [1 nChans*nEvents]);
XX= zeros(N, nChans*nEvents);
for iw= 1:nWindows,
    XX= XX + abs(fft(dat.x(iv,:).*Win, N)).^2;
    iv= iv + opt.step;
end

XX = XX/(nWindows*normWin^2);
dat.x= reshape( 10*log10( XX(bInd,:)+eps ), [length(bInd), nChans, nEvents]);
dat.yUnit= 'dB';
dat.t= Freq;
dat.xUnit= 'Hz';


%%
% nB=size(band);
% Freq=(0:N/2)*fs/N;
% bInd=[2: 41];
% Sf=band(1);
% Ef=band(2);
% 
% Freq=Freq(Sf+1:Ef+1);
% 
% XX= zeros(N, nChans*nEvents);
% 
% iv= 1:min(N, T);
% Win= repmat(Win(:), [1 nChans*nEvents]);
% nWindows= 1 + max(0, floor((T-N)/step));
% 
% for iw= 1:nWindows,
%             XX= XX + abs(fft(dat.x(iv,:).*Win, N)).^2;
%             iv= iv + step;
%         end
%         XX = XX/(nWindows*normWin^2);
%         dat.x= reshape( 10*log10( XX(bInd,:)+eps ), [length(bInd), nChans, nEvents]);
%         dat.yUnit= 'dB';
% 
% dat.t= Freq;
% dat.xUnit= 'Hz';
% 
% end






% opt = opt_cellToStruct(varargin{:});
% opt_base={'win',''; 'N',''; 'step','' ; 'scale',''};
% opt_base=opt_cellToStruct(opt_base);




% 
% strcmp(opt,opt_base)
% % if strcmp(
% opt_default={'win',dat.fs; 'N',dat.fs; 'step',dat.fs/2 ; 'scale','db'};
%     %%
% opt_default=opt_cellToStruct(opt_default);
% 
% [T, nEvents , nChans]= size(dat.x);
% 
% if isempty(dat)
%     warning('Warning! data is empty');
% end
% if isempty(ban)
%     warning('Band is not exist.');
% end
% 
% if ~length(opt.win)==1 %%opt의 윈도우가 비면
%     if opt.win>T
%         warning('Window size is longer than size');   
%       end
%      opt.win=ones(dat.fs,1)
% end

% if length(opt.win)==1 
%     opt.win=opt_default.win; 
% end
% if isemty(opt.N) opt.win=opt_default.N; end
% if isemty(opt.step) opt.step=opt_default.step; end
% if isemty(opt.scale) opt.scale=opt_default.scale; end



% %%
% N= length(opt.Win);
% normWin  = norm(opt.Win) ;
% 
% Win=ones(dat.fs); %%기본 윈도우 일때
% Win=ones  %(varagin에서 들어올때)
% [T, nEvents , nChans]= size(dat.x);
% normWin=norm(Win);

% %% 추가해야됨
% 
% 
% N= length(Win);
% spectrum
% 
% 
% fs=dat.fs;
% % dat.fs
% normWin=norm(N);
% N=100;
% step=N/2;
% 
% 
% nB=size(band);
% Freq=(0:N/2)*fs/N;
% bInd=[2: 41];
% Sf=band(1);
% Ef=band(2);
% 
% Freq=Freq(Sf+1:Ef+1);
% 
% XX= zeros(N, nChans*nEvents);
% 
% iv= 1:min(N, T);
% Win= repmat(Win(:), [1 nChans*nEvents]);
% nWindows= 1 + max(0, floor((T-N)/step));
% 
% for iw= 1:nWindows,
%             XX= XX + abs(fft(dat.x(iv,:).*Win, N)).^2;
%             iv= iv + step;
%         end
%         XX = XX/(nWindows*normWin^2);
%         dat.x= reshape( 10*log10( XX(bInd,:)+eps ), [length(bInd), nChans, nEvents]);
%         dat.yUnit= 'dB';
% 
% dat.t= Freq;
% dat.xUnit= 'Hz';
% 
% end

