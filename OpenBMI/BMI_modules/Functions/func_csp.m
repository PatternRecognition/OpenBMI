function [ dat_csp, CSP_W, CSP_D ] = func_csp( dat, varargin )
%PROC_CSP Summary of this function goes here
%   Detailed explanation goes here
% [SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
if nargin==0
    warning('Parameter is missing');
end

if ~isfield(dat,'x')
    error('Parameter is missing: dat.x')    
end

% Default parameters for CSP
opt_default={'nPatterns',3;
    'cov','normal';
    'score','eigenvalue';
    'policy','normal'};

if isempty(varargin)  % default parameter setting
    opt=opt_cellToStruct(opt_default)
else
    if iscell(varargin{:})
        opt=opt_cellToStruct(varargin{:});
    elseif isstruct(varargin{:}) % already structure(x-validation)
        opt=varargin{:}
    end
    for i=1:length(opt_default) % setting default parameters if some of input parameter is missing in varargin
        if ~isfield(opt,opt_default(i,1))
            opt.(opt_default{i,1})=opt_default{i,2};
        end
    end
end

%% default setting
dat_csp=dat;dat_csp.x=[];

%% calculate classwise covariance matrices
[nDat nTrials nChans]=size(dat.x);
nClasses=length(dat.class);
R= zeros(nChans, nChans, nClasses);

switch(lower(opt.cov))
    case 'normal'
        for i=1:nClasses
            idx= find(dat.y_logic(i,:));  %% ??? mrk.y�� ����
            tDat=zeros([nDat length(idx) nChans]);
            tDat= dat.x(:,idx,:);
            tDat=reshape(tDat, [nDat*length(idx),nChans]);
            R(:,:,i)=cov(tDat);
        end
        
    case 'average' %%malfunction
        %         for c= 1:nClasses,
        %             C= zeros(nChans, nChans);
        %             idx= find(epo.y_logical(c,:));
        %             for m= idx,
        %                 C= C + cov(squeeze(epo.x(:,m,:)));
        %             end
        %             R(:,:,c)= C/length(idx);
        %         end
    otherwise,
        error('check the cov options')
end
% R(isnan(R))=0;
[W,D]= eig(R(:,:,2),R(:,:,1)+R(:,:,2));

switch(lower(opt.score))
    case 'eigenvalue'
        score=diag(D);
end

nPattern=opt.nPatterns;
switch opt.policy
    case 'normal'
        CSP_W = W( :, [1:nPattern, end-nPattern+1:end] );
        CSP_D = score([1:nPattern, end-nPattern+1:end]);
    case 'directorscut'
        absscore= 2*(max(score, 1-score)-0.5);
        [dd,di]= sort(score);
        Nh= floor(nChans/2);
        iC1= find(ismember(di, 1:Nh,'legacy'));
        iC2= flipud(find(ismember(di, [nChans-Nh+1:nChans],'legacy')));
        iCut= find(absscore(di)>=0.66*max(absscore));
        idx1= [iC1(1); intersect(iC1(2:nPattern), iCut,'legacy')];
        idx2= [iC2(1); intersect(iC2(2:nPattern), iCut,'legacy')];
        fi= di([idx1; flipud(idx2)]);
        CSP_W = W(:,fi);
        CSP_D = score(fi);
end

dat=func_projection(dat, CSP_W);
dat_csp.x=dat.x;

% stack
if isfield(dat_csp, 'stack')
    c = mfilename('fullpath');
    c = strsplit(c,'\');
    dat_csp.stack{end+1}=c{end};
end
end

