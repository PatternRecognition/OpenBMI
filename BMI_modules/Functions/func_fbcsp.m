function [ filters ] = func_fbcsp( Dat, varargin )
%FUNC_FBCSP Summary of this function goes here
%   Detailed explanation goes here
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

%% Specral filtering
[nDat, nTri, nCH] = size( C1.x ); % same size of [dat, trial, ch] for each class
X1 = reshape( C1.x, [nDat*nTri,nCH ] );
X2 = reshape( C2.x, [nDat*nTri,nCH ] );

x_flt=cell(2, length(opt.Filters)) % binary, 
for i=1:length(opt.Filters)
    x_flt{1, i}= prep_filter(X1,{'frequency', opt.Filters(i,:); 'fs', opt.fs});
    x_flt{2, i}= prep_filter(X2,{'frequency', opt.Filters(i,:); 'fs', opt.fs});
end

%% Spatial filtering
CSPFilter = cell( 1, length(opt.Filters) );
for i=1:length(opt.Filters)
    X1 = x_flt{1, i};
    X2 = x_flt{2, i};
    
    %% Here is basic CSP filtering.
    S1 = cov(X1(:,:));    
    S2 = cov(X2(:,:));    
    [W,D] = eig(S1, S1+S2);
    %%
    CSPFilter{i}.W = W( :, [1:opt.numCSPPatterns, end - opt.numCSPPatterns+1:end] );
    Dd = diag(D);
    CSPFilter{i}.D = Dd([1:opt.numCSPPatterns, end - opt.numCSPPatterns+1:end]);
end

%% Feature extraction

features=cell(2, length(opt.Filters))
for i=1:length(opt.Filters)    
    for j=1:2 % for two classes
        dat=func_projection(x_flt{j,i}, CSPFilter{i}.W);
        dat=reshape( dat, [nDat,nTri,size(dat,2)]);  
        features{j,i} = squeeze( log(var(dat, 0, 1)) ); % log-variance feature
    end
end

%% Mutual information
kernelWidth=1;
for i=1:length(opt.Filters)
    miValue(i) = proc_mutual_information( features{1, i}, features{2, i}, kernelWidth );
end

end
















