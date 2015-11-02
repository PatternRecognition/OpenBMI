function [ epo_csp, CSP_W, CSP_D ] = func_csp( epo, varargin )
%PROC_CSP Summary of this function goes here
%   Detailed explanation goes here

if ~varargin{end}
    varargin=varargin{1,1}; %cross-validation procedures
end;

if ~length(varargin)
    opt_input=[];
else
    opt_input=opt_proplistToCell(varargin{:});
end
%% default setting
epo_csp=epo;epo_csp.x=[];
if isfield(epo,'chan')
    epo_csp.origChan=epo.chan;
end
epo_csp.chan=[];
opt_default={'nPatterns',3;'cov','normal';'score','eigenvalue';'policy','normal'};

%Change input parameters
for i=1:size(opt_input,1)
    [a b]=find(strcmpi(opt_default,opt_input{i}));
    if b
        opt_default{a,2}=opt_input{i,2};
    end
end

for i=1:length(opt_default)
    fid=opt_default{i,1};
    opt.(fid)=opt_default{i,2};
end

%% calculate classwise covariance matrices
epo.x=epo.x(2:end,:,:);
[nDat nTrials nChans]=size(epo.x);
nClasses=length(epo.class);
R= zeros(nChans, nChans, nClasses);

switch(lower(opt.cov))
    case 'normal'
        for i=1:nClasses
            idx= find(epo.y_logical(i,:));
            dat= epo.x(:,idx,:);
            dat=reshape(dat, [nDat*length(idx),nChans]);
            R(:,:,i)=cov(dat);
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

nPattern=str2num(opt.nPatterns);
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

dat=func_projection(epo, CSP_W);
epo_csp.x=dat.x;

% stack
c = mfilename('fullpath');
c = strsplit(c,'\');
epo_csp.stack{end+1}=c{end};

end

