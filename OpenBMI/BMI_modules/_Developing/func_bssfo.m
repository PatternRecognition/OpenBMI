function [ updatedBSSFO ] = func_bssfo( Dat, varargin )
%FUNC_BSSFO Summary of this function goes here
%   Detailed explanation goes here
opt=opt_cellToStruct(varargin{:});

if isfield(Dat, 'fs')
    opt.fs=Dat.fs;
elseif sifield(opt, 'fs')
    
else
    error('OpenBMI: "fs" is missing')
end

%% Initial frequency band
initSMC.numBands = opt.numBands*length(opt.frequency);

tmp=[],tmp2=[];
for i=1:length(opt.frequency)
    mu=opt.frequency{i};
    stdv=opt.std{i};
    sigma=[stdv 0; 0 stdv];
    tmp=mvnrnd(mu,sigma,opt.numBands);
    tmp2=cat(1, tmp2, tmp);
end

initSMC.sample=tmp2';

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
for i=1:opt.numIteration
    fprintf( '%d-th iteration...\n', i );
    if i>1
        oldBSSFO = proc_resamplingBSSFO( updatedBSSFO );
        oldBSSFO = proc_driftBSSFO( oldBSSFO );
    else
        oldBSSFO = initBSSFO;
    end
    verbose=1;
    [updatedBSSFO, x_flt, CSP, features] = measurementBSSFO( C1.x, C2.x, opt.fs, oldBSSFO, opt.numCSPPatterns, verbose );
    bar(updatedBSSFO.miValue)
end

% Determine filter banks based on the updatedBSSFO sample weights
[val, idx] = sort( updatedBSSFO.weight, 'descend' );
updatedBSSFO.weight = val;
updatedBSSFO.sample = updatedBSSFO.sample(:, idx);

end

