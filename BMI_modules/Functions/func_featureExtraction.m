function [ out ] = func_featureExtraction( in, varargin )
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

if isstruct(in)
    dat=in.x;
else
    dat=in;
end

switch lower(param{1})
    case 'logvar'
        dat=squeeze(log(var(dat)));
        dat=dat'
        
    case 'erpmean'
        [nDat, nTrials, nChans]= size(dat);
        T=param{2};
        nMeans= round(nDat/T);
        dat_=zeros(nMeans, nTrials, nChans);
        for i=1:nMeans
            if i==nMeans
                temp=mean(dat((i*T-T)+1:end,:,:),1);
            else
                temp=mean(dat((i*T-T)+1:i*T,:,:),1);
            end           
            dat_(i,:,:)=temp; temp=[];
        end
         [nDat, nTrials, nChans]= size(dat_);       
%          dat=dat_;
         dat= reshape(permute(dat_,[1 3 2]), [nDat*nChans nTrials]);        
end


if isstruct(in)
    in.x=dat;
else
    in=dat;
end
out=in;

if isfield(in,'stack') %% put in the function
    % stack
    c = mfilename('fullpath');
    c = strsplit(c,'\');
    in.stack{end+1}=c{end};
end
end

