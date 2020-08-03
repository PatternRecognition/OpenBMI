function [cnt,values] = proc_removeEyemoves(cnt, cntarti, mrkarti, options)
%[cnt,values] = proc_removeEyemoves(cnt, cntarti, mrkarti, options)
%
% cnt is the real EEG
% cntarti are the EEG of the artifact measurement
% mrkarti are the markes in the artifact measurement
% options is a struct with the following entries
%        channels: a cell array with channels in the usual format
%        for make_Segments, where the algorithm only makes PCA. You
%        get then only this channels as ocular potentials back
%        (default all)
%        moves: a word with the following letters + numbers (default 'v1h1bio')
%        l# : take eye movements to the left (# number of
%        SourceVectors, default 1)
%        r# : take eye movements to the right (")
%        u# : take eye movements up             (")
%        d# : take eye movements down(")
%        v# : combine up and down (lr then automatical is set)(")
%        h# : combine left and right (ud then automatical is
%        set)(")
% if v or h are, the numbers behind l,r,u,d are ignored
%        s : take the starting point of the interval
%        e : take the end point of the interval
%        b : =se
%        i : negate the right and down signals for getteing
%        SourceVectors (only use if v or h is set)
%        o : the signal to the end belong to the other direction
%
%        method: a method to find the blinks (default:
%        readSamples_simple) (first argument: cntarti, second
%        argument: intervals, range will be given as field in the
%        third argument
%        methodoptions: options for the call of the method (third
%        argument) (default [], readSamples_simple has senseful
%        default (see there for documnetation))
%        Note: all values (with defautls here) are set here.
%        chan: .v channel for vertical movement (EOGv)
%              .h channel for horizontal movement (EOGh)
%        lowpass: a frequenzband for lowpassfiltering (of the
%        artifcat measurement (default: no lowpass)
%        range: a range around the meanpoint of the movement (default:
%        [-100 500] 
% options can be a vector (2-dim) and then it is the range
%        
% Output: fv, the corrected EEG
%         values struct with the following entries
%         variance: proportion of the variance 
%         SV: the SOurceVectors as matrice
%         SW: the SourceWaveforms as matrice
%
% Guido Dornhege
% 04.04.02

defaultrange = [-100 500];
defaultchannel = [];
defaultmoves = 'v1h1bio';
defaultmethod = 'readSamples_simple';
defaultmethodoptions = [];
defaultlowpass = [];
defaultnumber = 1;

% check input
if nargin<3 
  error('not enough input arguments');
end
  
if ~exist('options') | isempty(options)
  options = defaultrange;
end
  
if ~isstruct(options)
  options.range = options;
end

if ~isfield(options,'range')
  options.range = defaultrange;
end

if ~isfield(options,'channels')
  options.channels = defaultchannel;
end

if ~isfield(options,'moves')
  options.moves = defaultmoves;
end

if ~isfield(options,'method')
  options.method = defaultmethod ;
end

if ~isfield(options,'methodoptions')
  options.methodoptions = defaultmethodoptions;
end

if ~isfield(options,'lowpass')
  options.lowpass = defaultlowpass ;
end

% lowpass
if ~isempty(options.lowpass)
  cntarti = proc_filtByFFT(cntarti,options.lowpass);
end

if ~isfield(options,'chan')
  options.chan.v ='EOGv';
  options.chan.h = 'EOGh';
end

if ~isstruct(options.chan)
  options.chan.v = options.chan;
  options.chan.h = options.chan.v;
end

if ~isfield(options.chan,'v')
  options.chan.v = 'EOGv';
end

if ~isfield(options.chan,'h')
  options.chan.v = 'EOGh';
end

% now get the Aritfacts
opt.same = 0;
opt.channels = options.channels;


% prepare get_ArtifactsPCA
horizontal = [];
left = [];
right = [];
c = strfind(options.moves,'h');
if ~isempty(c)
  c = c(1);
  j = 0;
  while c+j+1<= length(options.moves) & ~isempty(str2num(options.moves((0:j)+c+1))) & isreal(str2num(options.moves((0:j)+c+1)))
    j = j+1;
  end
  if j == 0
    number = defaultnumber;
  else
    number = str2num(options.moves((1:j)+c));
  end
  
  horizontal.name = 'horizontal';
  horizontal.artifacts = {'Augen links', 'Augen rechts'};
  horizontal.number = number;
  horizontal.readSamples = options.method;
  horizontal.readOptions = options.methodoptions;
  horizontal.readOptions.range = options.range;
  
  if ~isempty(strfind(options.moves,'b'))
    horizontal.readOptions.number = 2;
    if ~isempty(strfind(options.moves,'o'))
      horizontal.readOptions.factors = [1 -1];
    else
      horizontal.readOptions.factors = [1 1]; 
    end
  elseif ~isempty(strfind(options.moves,'s'))
    horizontal.readOptions.number = 1;
    horizontal.readOptions.factors = 1 ;
  elseif ~isempty(strfind(options.moves,'e'))
    horizontal.readOptions.number = 2;
    horizontal.readOptions.factors = [0 1] ;
  else
    horizontal.readOptions.number = 2;
    if ~isempty(strfind(options.moves,'o'))
      horizontal.readOptions.factors = [1 -1];
    else
      horizontal.readOptions.factors = [1 1]; 
    end   
  end
  
  if ~isempty(strfind(options.moves,'i')) 
    horizontal.readOptions.factorsartifacts = [1 -1];
  else
    horizontal.readOptions.factorsartifacts = [1 1];    
  end
  
else 
  c = strfind(options.moves,'l');
  if ~isempty(c)
    c = c(1);
    j = 0;
    while c+j+1<= length(options.moves) & ~isempty(str2num(options.moves((0:j)+c+1)))& isreal(str2num(options.moves((0:j)+c+1)))
      j = j+1;
    end
    if j == 0
      number = defaultnumber;
    else
      number = str2num(options.moves((1:j)+c));
    end
    
    left.name = 'left';
    left.artifacts = 'Augen links';
    left.number = number;
    left.readSamples = options.method;
    left.readOptions = options.methodoptions;
    left.readOptions.range = options.range;
    
    if ~isempty(strfind(options.moves,'b'))
      left.readOptions.number = 2;
      if ~isempty(strfind(options.moves,'o'))
	left.readOptions.factors = [1 -1];
      else
	left.readOptions.factors = [1 1]; 
      end
    elseif ~isempty(strfind(options.moves,'s'))
      left.readOptions.number = 1;
      left.readOptions.factors = 1 ;
    elseif ~isempty(strfind(options.moves,'e'))
      left.readOptions.number = 2;
      left.readOptions.factors = [0 1] ;
    else
      left.readOptions.number = 2;
      if ~isempty(strfind(options.moves,'o'))
	left.readOptions.factors = [1 -1];
      else
	left.readOptions.factors = [1 1]; 
      end   
    end
  end
  c = strfind(options.moves,'r');
  if ~isempty(c)
    c = c(1);
    j = 0;
    while c+j+1<= length(options.moves) & ~isempty(str2num(options.moves((0:j)+c+1)))& isreal(str2num(options.moves((0:j)+c+1)))
      j = j+1;
    end
    if j == 0
      number = defaultnumber;
    else
      number = str2num(options.moves((1:j)+c));
    end
    
    right.name = 'right';
    right.artifacts = 'Augen rechts';
    right.number = number;
    right.readSamples = options.method;
    right.readOptions = options.methodoptions;
    right.readOptions.range = options.range;
    
    if ~isempty(strfind(options.moves,'b'))
      right.readOptions.number = 2;
      if ~isempty(strfind(options.moves,'o'))
	right.readOptions.factors = [1 -1];
      else
	right.readOptions.factors = [1 1]; 
      end
    elseif ~isempty(strfind(options.moves,'s'))
      right.readOptions.number = 1;
      right.readOptions.factors = 1 ;
    elseif ~isempty(strfind(options.moves,'e'))
      right.readOptions.number = 2;
      right.readOptions.factors = [0 1] ;
    else
      right.readOptions.number = 2;
      if ~isempty(strfind(options.moves,'o'))
	right.readOptions.factors = [1 -1];
      else
	right.readOptions.factors = [1 1]; 
      end   
    end
  end
end


      



vertical = [];
up =[];
down = [];
c = strfind(options.moves,'v');
if ~isempty(c)
  c = c(1);
  j = 0;
  while c+j+1<= length(options.moves) & ~isempty(str2num(options.moves((0:j)+c+1)))& isreal(str2num(options.moves((0:j)+c+1)))
    j = j+1;
  end
  if j == 0
    number = defaultnumber;
  else
    number = str2num(options.moves((1:j)+c));
  end
  
  vertical.name = 'vertical';
  vertical.artifacts = {'Augen hoch', 'Augen runter'};
  vertical.number = number;
  vertical.readSamples = options.method;
  vertical.readOptions = options.methodoptions;
  vertical.readOptions.range = options.range;

  if ~isempty(strfind(options.moves,'b'))
    vertical.readOptions.number = 2;
    if ~isempty(strfind(options.moves,'o'))
      vertical.readOptions.factors = [1 -1];
    else
      vertical.readOptions.factors = [1 1]; 
    end
  elseif ~isempty(strfind(options.moves,'s'))
    vertical.readOptions.number = 1;
    vertical.readOptions.factors = 1 ;
  elseif ~isempty(strfind(options.moves,'e'))
    vertical.readOptions.number = 2;
    vertical.readOptions.factors = [0 1] ;
  else
    vertical.readOptions.number = 2;
    if ~isempty(strfind(options.moves,'o'))
      vertical.readOptions.factors = [1 -1];
    else
      vertical.readOptions.factors = [1 1]; 
    end   
  end
  
  if ~isempty(strfind(options.moves,'i')) 
    vertical.readOptions.factorsartifacts = [1 -11];
  else
    vertical.readOptions.factorsartifacts = [1 1];    
  end
  
else 
  c = strfind(options.moves,'u');
  if ~isempty(c)
    c = c(1);
    j = 0;
    while c+j+1<= length(options.moves) & ~isempty(str2num(options.moves((0:j)+c+1)))& isreal(str2num(options.moves((0:j)+c+1)))
      j = j+1;
    end
    if j == 0
      number = defaultnumber;
    else
      number = str2num(options.moves((1:j)+c));
    end
    
    up.name = 'up';
    up.artifacts = 'Augen hoch';
    up.number = number;
    up.readSamples = options.method;
    up.readOptions = options.methodoptions;
    up.readOptions.range = options.range;
    
    if ~isempty(strfind(options.moves,'b'))
      up.readOptions.number = 2;
      if ~isempty(strfind(options.moves,'o'))
	up.readOptions.factors = [1 -1];
      else
	up.readOptions.factors = [1 1]; 
      end
    elseif ~isempty(strfind(options.moves,'s'))
      up.readOptions.number = 1;
      up.readOptions.factors = 1 ;
    elseif ~isempty(strfind(options.moves,'e'))
      up.readOptions.number = 2;
      up.readOptions.factors = [0 1] ;
    else
      up.readOptions.number = 2;
      if ~isempty(strfind(options.moves,'o'))
	up.readOptions.factors = [1 -1];
      else
	up.readOptions.factors = [1 1]; 
      end   
    end
  end     
  c = strfind(options.moves,'d');
  if ~isempty(c)
    c = c(1);
    j = 0;
    while c+j+1<= length(options.moves) & ~isempty(str2num(options.moves((0:j)+c+1)))& isreal(str2num(options.moves((0:j)+c+1)))
      j = j+1;
    end
    if j == 0
      number = defaultnumber;
    else
      number = str2num(options.moves((1:j)+c));
    end
    
    down.name = 'down';
    down.artifacts = 'Augen runter';
    down.number = number;
    down.readSamples = options.method;
    down.readOptions = options.methodoptions;
    down.readOptions.range = options.range;
    
    if ~isempty(strfind(options.moves,'b'))
      down.readOptions.number = 2;
      if ~isempty(strfind(options.moves,'o'))
	down.readOptions.factors = [1 -1];
      else
	down.readOptions.factors = [1 1]; 
      end
    elseif ~isempty(strfind(options.moves,'s'))
      down.readOptions.number = 1;
      down.readOptions.factors = 1 ;
    elseif ~isempty(strfind(options.moves,'e'))
      down.readOptions.number = 2;
      down.readOptions.factors = [0 1] ;
    else
      down.readOptions.number = 2;
      if ~isempty(strfind(options.moves,'o'))
	down.readOptions.factors = [1 -1];
      else
	down.readOptions.factors = [1 1]; 
      end   
    end
  end
end

if ~isempty(options.channels)
  cnt = proc_selectChannels(cnt, options.channels{:});
end


if ~isempty(horizontal)
  if nargin<2
    horizontal  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, horizontal);
  else
    [horizontal,value]  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, horizontal);
    values.horizontal.SW = value.SourceWaveforms{1};
    values.horizontal.variance = value.values;
    values.horizontal.SV = value.SourceVectors;
  end
  cnt.x = cnt.x - horizontal.x;
end

if ~isempty(left)
  if nargin<2
    left  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, left);
  else
    [left,value]  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, left);
    values.left.SW = value.SourceWaveforms{1};
    values.left.variance = value.values;
    values.left.SV = value.SourceVectors;
  end
  cnt.x = cnt.x - left.x;
end

if ~isempty(right)
  if nargin<2
    right  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, right);
  else
    [right,value]  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, right);
    values.right.SW = value.SourceWaveforms{1};
    values.right.variance = value.values;
    values.right.SV = value.SourceVectors;
  end
cnt.x = cnt.x - right.x;
end

if ~isempty(up)
  if nargin<2
    up  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, up);
  else
    [up,value]  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, up);
    values.up.SW = value.SourceWaveforms{1};
    values.up.variance = value.values;
    values.up.SV = value.SourceVectors;
  end
cnt.x = cnt.x - up.x;
end

if ~isempty(down)
  if nargin<2
    down  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, down);
  else
    [down,value]  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, down);
    values.down.SW = value.SourceWaveforms{1};
    values.down.variance = value.values;
    values.down.SV = value.SourceVectors;
  end
  cnt.x = cnt.x - down.x;
end

if ~isempty(vertical)
  if nargin<2
    vertical  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, vertical);
  else
    [vertical,value]  = get_ArtifactsPCA(cnt, cntarti, mrkarti, opt, vertical);
    values.vertical.SW = value.SourceWaveforms{1};
    values.vertical.variance = value.values;
    values.vertical.SV = value.SourceVectors;
  end
  cnt.x = cnt.x - vertical.x;
end
