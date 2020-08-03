function dat= proc_spectrum(dat, band, varargin)
%dat= proc_spectrum(dat, band, <win/N, step>)
%dat= proc_spectrum(dat, band, <opts>)
%
% calculate the power spectrum
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

% bb, ida.first.fhg.de

if ~isempty(varargin) && isnumeric(varargin{1}),
    %% arguments given as <win/N, step>
    opt.win= varargin{1};
    if length(varargin)>=2 && isnumeric(varargin{2}),
        opt.step= varargin{2};
    end
else
    %% arguments given as <opt>
    opt= propertylist2struct(varargin{:});
end
[opt, isdefault]= ...
    set_defaults(opt, ...
    'win', dat.fs, ...
    'scaling', 'db');

if isfield(opt, 'db_scaled'),
    if ~isdefault.scaling,
        error('not both properties *scaling* and *db_scaled* can be defined');
    end
    if opt.db_scaled,
        opt.scaling= 'db';
    else
        opt.scaling= 'unnormalized';
    end
end

[T, nChans, nEvents]= size(dat.x);
if length(opt.win)==1,
    if opt.win>T,
        if ~isdefault.win,
            warning(['Requested window length longer than signal: ' ...
                'shortening window, no zero-padding!']);
        end
        opt.win= T;
    end
    opt.win=ones(opt.win,1);
end
N= length(opt.win);
normWin  = norm(opt.win) ;
if ~isfield(opt, 'step'), opt.step= floor(N/2); end
if ~exist('band','var'), band= [0 dat.fs/2]; end

[bInd, Freq]= getBandIndices(band, dat.fs, N);
XX= zeros(N, nChans*nEvents);
nWindows= 1 + max(0, floor((T-N)/opt.step));
iv= 1:min(N, T);
Win= repmat(opt.win(:), [1 nChans*nEvents]);

switch(lower(opt.scaling)),
    case 'db',
        for iw= 1:nWindows,
            XX= XX + abs(fft(dat.x(iv,:).*Win, N)).^2;
            iv= iv + opt.step;
        end
        XX = XX/(nWindows*normWin^2);
        dat.x= reshape( 10*log10( XX(bInd,:)+eps ), [length(bInd), nChans, nEvents]);
        dat.yUnit= 'dB';
    case 'power',
        for iw= 1:nWindows,
            XX= XX + abs(fft(dat.x(iv,:).*Win, N).^2);
            iv= iv + opt.step;
        end
        dat.x= reshape(XX(bInd,:)/(nWindows*normWin^2), [length(bInd), nChans, nEvents]);
        dat.yUnit= 'power';
    case 'unnormalized',
        for iw= 1:nWindows,
            XX= XX + abs(fft(dat.x(iv,:).*Win, N).^2);
            iv= iv + opt.step;
        end
        dat.x= reshape(XX(bInd,:)/nWindows, [length(bInd), nChans, nEvents]);
        dat.yUnit= 'power';
    case 'normalized',
        for iw= 1:nWindows,
            XX= XX + abs(fft(dat.x(iv,:).*Win, N));
            iv= iv + opt.step;
        end
        XX= XX*2/N;
        dat.x= reshape(XX(bInd,:)/nWindows, [length(bInd), nChans, nEvents]);
    case 'complex',
        for iw= 1:nWindows,
            XX= XX + fft(dat.x(iv,:).*Win, N);
            iv= iv + opt.step;
        end
        XX = XX/(nWindows*normWin^2);
        dat.x= reshape( XX(bInd,:), [length(bInd), nChans, nEvents]);
        dat.yUnit= 'complex';
    otherwise,
        error('unknown choice for property *scaling*');
end

dat.t= Freq;
dat.xUnit= 'Hz';
