function [ out ] = func_featureExtraction( dat, varargin )
%PROC_FEAEXTRACTION Summary of this function goes here
%   Detailed explanation goes here
if ~varargin{end}
    varargin=varargin{1,1}; %cross-validation procedures
end;

if length(varargin)>1
 param=opt_proplistToCell(varargin{:});
else
    param=varargin;
end

if isstruct(dat)
    tDat=dat.x;
else
    tDat=dat;
end

switch lower(param{1})
    case 'logvar'
        tDat=squeeze(log(var(tDat)));
        tDat=tDat';
        
    case 'erpmean'
        [nDat, nTrials, nChans]= size(tDat);
        T=param{2};
        nMeans= round(nDat/T);
        dat_=zeros(nMeans, nTrials, nChans);
        for i=1:nMeans
            if i==nMeans
                temp=mean(tDat((i*T-T)+1:end,:,:),1);
            else
                temp=mean(tDat((i*T-T)+1:i*T,:,:),1);
            end           
            dat_(i,:,:)=temp; temp=[];
        end
         [nDat, nTrials, nChans]= size(dat_);       
%          dat=dat_;
         tDat= reshape(permute(dat_,[1 3 2]), [nDat*nChans nTrials]);        
end


if isstruct(dat)
    dat.x=tDat;
else
    dat=tDat;
end
out=dat;

if isfield(dat,'stack') %% put in the function
    % stack
    c = mfilename('fullpath');
    c = strsplit(c,'\');
    dat.stack{end+1}=c{end};
end
end

