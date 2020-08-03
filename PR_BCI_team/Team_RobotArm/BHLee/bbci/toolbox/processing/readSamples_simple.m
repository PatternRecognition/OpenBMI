function samp = readSamples_simple(cnt,iv,options)
%READSAMPLES_SIMPLE
% 
% Read Samples out of a given EEG in the intervals with the given
% options
% cnt: EEG
% iv: a intervalstructure you get from get_relevantArtifacts
% options a struct with the following entries (empty: all defaults,
% or range)
%      - range: range behind and before a given mark, where the
%      samples you get from (default [-100 500])
%      - number: a number of samples you want to find in an
%      interval (2, begin and end)
%      - onsetcorr: a string which can be 'grad', 'gradplus', 'gradminus',
%      'amp', 'ampmax', 'ampmin' or a function which calculates the
%      place where the onset of the artifact is.  ('grad') (first
%      argument is ever the datas (cnt) and the second
%      argument is ever the place of the assumed onset)
%      - onsetcorroptions: options for the functions above (for
%      grad and amp ranges before and after marks where it can be)
%      ([])
%      - factorsartifacts: factors for handling with different artifacts for
%      combination (1 .. 1)
%      - factors: factors for handling with the different points in
%      a time interval after separation (1 .. 1)
%
% Output: a three dimensional matrice, where in the rows are a time
% signal, in columns the channels, and in the third dimension are
% the samples
%
% Guido Dornhege
% 27.03.02

% check the input
if ~exist('options') | isempty(options)
  options = [-100 500];
end

if ~isstruct(options)
  options.range = options;
end

if ~isfield(options,'range')
  options.range = [-100 600];
end

if ~isfield(options,'number')
  options.number = 2;
end

if ~isfield(options,'onsetcorr')
  options.onsetcorr = 'grad';
end

if ischar(options.onsetcorr)
  options.onsetcorr = str2func(options.onsetcorr);
end

if ~isfield(options,'onsetcorroptions')
  options.onsetcorroptions = [];
end

if ~isfield(options,'factorsartifacts')
  options.factorsartifacts = ones(1,length(iv.artifacts));
end

if ~isfield(options,'factors')
  options.factors = ones(1,options.number);
end

% frequencies
options.range = union(round(options.range*cnt.fs/1000),[]);

if options.number<1 
  return
end

samp =[];
for arti =  1:length(iv.artifacts)
  sampl = [];
  intervals = iv.int{arti};
  for ival = 1:size(iv.int{arti},1)
      iva = intervals(ival,:);
      if options.number <2
	steps = iva(2)-iva(1)+1;
      elseif iva(1)==iva(2)
	steps = 1;
      else
	steps = (iva(2)-iva(1))/(options.number-1);
      end
      for step = iva(1):steps:iva(2)
        pl = feval(options.onsetcorr, cnt, step, options.onsetcorroptions);    
	
	sam = cnt.x(pl+(options.range(1):options.range(2)),:);
	  
	
        sam = sam*options.factors(find(step==iva(1):steps:iva(2)));
	sampl = cat(3,sampl,sam);
      end
  end
  samp = cat(3,samp,sampl*options.factorsartifacts(arti));
end

	



%----------------------------------------
% SUBROUTINES
%----------------------------------------

function pl = grad(dat, onset, channels, x)

if ~exist('channels') | isempty(channels)
  channels = ones(1,size(dat.x,2));
elseif ischar(channels)
  c = find(strcmp(dat.clab,channels));
  channels = zeros(1,size(dat.x,2)); 
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==1
  c = chanind(dat.clab,channels{:});
  channels = zeros(1,size(dat.x,2));
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==2
  chan = zeros(1,size(dat.x,2));
  for i = 1:size(channels,2)
    c = chanind(dat.clab,channels{1,i});
    chan(c) = channels{2,i};
  end
  channels = chan;
end

if ~exist('x') | isempty(x)
  x = [0 1];
else 
  x = x*dat.fs/1000;
end  

[dummy,pl] = max(abs(dat.x(onset+(x(1)+1:x(2)),:)-dat.x(onset+(x(1):x(2)- ...
						  1),:))*channels');
pl = pl+onset;


function pl = gradplus(dat, onset, channels, x)
if ~exist('channels') | isempty(channels)
  channels = ones(1,size(dat.x,2));
elseif ischar(channels)
  c = find(strcmp(dat.clab,channels));
  channels = zeros(1,size(dat.x,2)); 
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==1
  c = chanind(dat.clab,channels{:});
  channels = zeros(1,size(dat.x,2));
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==2
  chan = zeros(1,size(dat.x,2));
  for i = 1:size(channels,2)
    c = chanind(dat.clab,channels{1,i});
    chan(c) = channels{2,i};
  end
  channels = chan;
end
if ~exist('x') | isempty(x)
  x = [0 1];
else 
  x = x*dat.fs/1000;
end  
[dummy,pl] = max((dat.x(onset+(x(1)+1:x(2)),:)-dat.x(onset+(x(1):x(2)- ...
						  1),:))*channels');
pl = pl+onset;


function pl = gradminus(dat,onset, channels,x)
if ~exist('channels') | isempty(channels)
  channels = ones(1,size(dat.x,2));
elseif ischar(channels)
  c = find(strcmp(dat.clab,channels));
  channels = zeros(1,size(dat.x,2)); 
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==1
  c = chanind(dat.clab,channels{:});
  channels = zeros(1,size(dat.x,2));
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==2
  chan = zeros(1,size(dat.x,2));
  for i = 1:size(channels,2)
    c = chanind(dat.clab,channels{1,i});
    chan(c) = channels{2,i};
  end
  channels = chan;
end

if ~exist('x') | isempty(x)
  x = [0 1];
else 
  x = x*dat.fs/1000;
end  
[dummy,pl] = min((dat.x(onset+(x(1)+1:x(2)),:)-dat.x(onset+(x(1):x(2)- ...
						  1),:))*channels');
pl = pl+onset;


function pl = amp(dat, onset,channels, x)
if ~exist('channels') | isempty(channels)
  channels = ones(1,size(dat.x,2));
elseif ischar(channels)
  c = find(strcmp(dat.clab,channels));
  channels = zeros(1,size(dat.x,2)); 
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==1
  c = chanind(dat.clab,channels{:});
  channels = zeros(1,size(dat.x,2));
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==2
  chan = zeros(1,size(dat.x,2));
  for i = 1:size(channels,2)
    c = chanind(dat.clab,channels{1,i});
    chan(c) = channels{2,i};
  end
  channels = chan;
end
if ~exist('x') | isempty(x)
  x = [0 0];
else 
  x = x*dat.fs/1000;
end  
[dummy,pl] = max(abs(dat.x(onset+(x(1):x(2)),:))*channels');
pl = pl+onset;

function pl = ampmax(dat,onset,channels,x)
if ~exist('channels') | isempty(channels)
  channels = ones(1,size(dat.x,2));
elseif ischar(channels)
  c = find(strcmp(dat.clab,channels));
  channels = zeros(1,size(dat.x,2)); 
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==1
  c = chanind(dat.clab,channels{:});
  channels = zeros(1,size(dat.x,2));
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==2
  chan = zeros(1,size(dat.x,2));
  for i = 1:size(channels,2)
    c = chanind(dat.clab,channels{1,i});
    chan(c) = channels{2,i};
  end
  channels = chan;
end
if ~exist('x') | isempty(x)
  x = [0 0];
else 
  x = x*dat.fs/1000;
end  
[dummy,pl] = max(dat.x(onset+(x(1):x(2)),:)*channels');
pl = pl+onset;


function pl = ampmin(dat,onset,channels,x)
if ~exist('channels') | isempty(channels)
  channels = ones(1,size(dat.x,2));
elseif ischar(channels)
  c = find(strcmp(dat.clab,channels));
  channels = zeros(1,size(dat.x,2)); 
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==1
  c = chanind(dat.clab,channels{:});
  channels = zeros(1,size(dat.x,2));
  channels(c) = 1;
elseif iscell(channels) & size(channels,1) ==2
  chan = zeros(1,size(dat.x,2));
  for i = 1:size(channels,2)
    c = chanind(dat.clab,channels{1,i});
    chan(c) = channels{2,i};
  end
  channels = chan;
end
if ~exist('x') | isempty(x)
  x = [0 0];
else 
  x = x*dat.fs/1000;
end  
[dummy,pl] = min(dat.x(onset+(x(1):x(2)),:)*channels');
pl = pl+onset;












