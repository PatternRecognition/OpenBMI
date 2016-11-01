function [out,r]=func_cca(dat,refre,varargin)
%% obtaining r values by CCA function
% func_cca:
%     This function calculating correlation parameters based on canonical correlation analysis(CCA).  
%     Canonical correlation analysis is a method for calculating the relationships between two multivariate signal. 
%
%Example:
%     [out, r]=func_cca(SMT,[13,22,25],{'harmonic', 3})
%     [out, r]=func_cca(SMT.x,[13,22,25],{'harmonic', 2; 'fs',100})
% Input:
%     dat - Data structure, continuous or epoched (continuous)
%     refre - Making reference frequency
%
% Option:
%     harmonic - Number of harmonic frequncy
% 	  fs - Setting the data sampling rate
%     channel - Selecting specific channel
%
% Returns:
%     out - Calculating classification result of class which maximum r value  
%     r - Calculating r values 
%

%% channel must be segemented
% function proc_CCA(dat,time, frequency, harmonic)
%
if iscell(varargin{:})
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure
end
if ~isstruct(dat)
    if ~isfield(opt,'fs')
        error('[OpenBMI] : input dat need to fs');
    end
    
    if isempty(dat)
        warning('[OpenBMI] Warning! data is empty.');
    end
    
    if isempty(refre)
        warning('[OpenBMI] Warning! reference frequency is empty.');
    end
    
    if ~isfield(opt,'harmonic')
        warning('[OpenBMI]: parameter "harmonic" is missing')
        opt.Harmonic=2;
    end
    [nDat nTrial nCH]=size(dat);
else

    if ~isfield(opt,'Channel')
        warning('OpenBMI: parameter "Channel" is missing')
        opt.Channel={'Cz'};
    end
    
    opt.fs=dat.fs;
    out.Channel=opt.Channel;
    dat=prep_selectChannels(dat,{'Name',opt.Channel});
    
    ivaltime=dat.ival;
    out.ivaltime=ivaltime;
        
    %     timerange=(dat.ival(end)-dat.ival(1))/opt.fs*0.1; %% check
    [nDat  nTrial nCH]=size(dat.x);
    out.class=dat.class;
    opt.fs=dat.fs;
    out.originclass=dat.y_dec;
    
    dat=dat.x
end

out.refFreq=refre;


if refre(end)*opt.Harmonic > opt.fs/2
    warning('OpenBMI: harmonic frequency cannot show. Harmonic is setting as 2')
    opt.harmonic=2;
end
out.harmonic=opt.Harmonic;


%% calculate time ranges for generating harmonic reference frequencies
t=[0:nDat*opt.fs*0.0001/(nDat-1):nDat*opt.fs*0.0001];

%% generate harmonic frequency
for i=1:length(refre)
    Y{i}=[sin(2*pi*refre(i)*t);cos(2*pi*refre(i)*t); sin(2*pi*2*refre(i)*t);cos(2*pi*2*refre(i)*t);sin(2*pi*3*refre(i)*t); cos(2*pi*3*refre(i)*t)];
    Y{i}=Y{i}(1:2*opt.Harmonic,:);
end

r=zeros(nTrial,length(Y));
%% calulate CCA parameter
for i=1:nTrial
    temp=dat(:,i); % channel
    temp=squeeze(temp);
    for j=1:length(Y)
        [A B cor_r{j}]=canoncorr(temp,Y{j}');
    end
    for k=1:length(Y)
        r(i,k)=[(cor_r{k})]; %%
    end
    [k, j]=max(r(i,:));
    out.class{i}=j;
end
out.class=out.class'
