function out= proc_bipolarChannels(dat, varargin)
%out= proc_bipolarChannels(dat, channel_list)
%
% IN  dat          - struct of continuous or epoched data
%     channel_list - list or cell array of string, defining the bipolar
%                    channels, e.g. {'C3-C4', CP3-CP4'}

if iscell(varargin{1}),
  if length(varargin)>1,
    error('either or');
  end
  bipo=varargin{1};
else
  bipo=varargin;
end

out= copyStruct(dat, 'x','clab');
[T nChans, nEpos]= size(dat.x);
nBipos= length(bipo);
out.x= zeros(T, nBipos, nEpos);
out.clab= cell(1, nBipos);
for ib= 1:nBipos,
  biStr= bipo{ib};
  is= find(biStr=='-');
  if isempty(is),
    error('each entry of the channel_list must contain a ''-'' sign');
  end
  chp= chanind(dat, biStr(1:is-1));
  chn= chanind(dat, biStr(is+1:end));
  out.x(:,ib,:)= dat.x(:,chp,:) - dat.x(:,chn,:);
  out.clab{ib}= [dat.clab{chp} '-' dat.clab{chn}];
end
