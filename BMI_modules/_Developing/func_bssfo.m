function [ updatedBSSFO ] = func_bssfo( Dat, varargin )
%FUNC_BSSFO Summary of this function goes here
%   Detailed explanation goes here
% [FilterBand]=func_bssfo(SMT, {'classes', {'right', 'left'};'frequency', {[7 15],[14 30]}; 'std', {5, 25}; ...
% 'numBands', 30; 'numCSPPatterns', 2; 'numIteration', 30});
% Dat: SMT epoched structure or data, in the case of binary data, the
% 'fs' option is needed

opt=opt_cellToStruct(varargin{:});
if isstruct(Dat)
    if isfield(Dat, 'fs')
        opt.fs=Dat.fs;
    elseif sifield(opt, 'fs')
        
    else
        error('OpenBMI: "fs" is missing')
    end
else
    if ~isfield(Dat, 'fs')
        error('OpenBMI: "fs" is missing')
    end
end
%% Initial frequency band
% initSMC.numBands = opt.numBands*length(opt.frequency);
% 
% tmp=[],tmp2=[];
% for i=1:length(opt.frequency)
%     mu=opt.frequency{i};
%     stdv=opt.std{i};
%     sigma=[stdv 0; 0 stdv];
%     tmp=mvnrnd(mu,sigma,opt.numBands);
%     tmp2=cat(1, tmp2, tmp);
% end
% 
% initSMC.sample=tmp2';
% 
% for i=1:initSMC.numBands
%     initSMC.sample(:, i) = opt_checkValidity( initSMC.sample(:, i) );
% end

initSMC.numBands = 30;
% Generate samples from a mixture of Gaussian
N = initSMC.numBands;
stdv = 25;
numMix = 2;
mu = cat(2 , [8; 13] , [14; 30]);                %(d x 1 x M)
mu = reshape( mu, [2, 1, numMix] );
sigma = cat(2 , [stdv 0; 0 stdv] , [stdv 0; 0 stdv] );   %(d x d x M)
sigma = reshape( sigma, [2 2 numMix] );
p = cat(2 , [0.5] , [0.5]);                       %(1 x 1 x M)
p = reshape( p, [1 1 numMix] );
[initSMC.sample , index] = sample_mvgm(N , mu , sigma , p);
for i=1:initSMC.numBands
   initSMC.sample(:, i) = opt_checkValidity( initSMC.sample(:, i) );
end

if iscell(Dat)  % cell type binary classes
    if length(Dat)==2
        C1=prep_selectClass(Dat, {'class', opt.classes{1}});
        C2=prep_selectClass(Dat, {'class', opt.classes{2}});
    else
        error('OpenBMI: The classes should be binary');
    end
else
    if isfield(opt, 'classes')
        if length(opt.classes) ~=2
            error('OpenBMI: The classes should be binary');
        else
            C1=prep_selectClass(Dat, {'class', opt.classes{1}});
            C2=prep_selectClass(Dat, {'class', opt.classes{2}});
        end
    end
end

C1.x=permute(C1.x, [3 1 2]);
C2.x=permute(C2.x, [3 1 2]);
initBSSFO=initSMC;

% band=[1 4;4 8;8 12;12 16;16 20;20 24;24 28;28 32;32 36;36 40;1 10;10 20;20 30;30 40;1 20;20 40; 1 40; 1 2 ]
% initBSSFO.numBands=length(band)
% initBSSFO.sample=band'

load initBand.mat
initBSSFO=initSMC
for i=1:opt.numIteration
    fprintf( '%d-th iteration...\n', i );
    if i>1
        oldBSSFO = proc_resamplingBSSFO( updatedBSSFO );
        oldBSSFO = proc_driftBSSFO( oldBSSFO );
    else
        oldBSSFO = initBSSFO;
    end
    verbose=1;
    [updatedBSSFO, x_flt, CSP, features] = measurementBSSFO2( C1.x, C2.x, opt.fs, oldBSSFO, opt.numCSPPatterns, verbose );
    bar(updatedBSSFO.miValue)
end

% Determine filter banks based on the updatedBSSFO sample weights
[val, idx] = sort( updatedBSSFO.weight, 'descend' );
updatedBSSFO.weight = val;
updatedBSSFO.sample = updatedBSSFO.sample(:, idx);

end

