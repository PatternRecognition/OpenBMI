function C = train_c45(xTr, yTr, varargin)

% function C= train_c45(xTr, yTr, nBoots, opt)
%           = train_c45(xTr, yTr, varargin)
%
% C= trainC45(xTr, yTr, <nBootstraps, opt/m>)
%
% Quinlan c4.5 decision tree
% 
% nBootstraps   are the number of iterations (positive number) (default:5)
% opt           is directly passed to c45, see documentation there
% 		.m : minimal number of objects in branch (default 2)
%               .v : verbosity (0-3) (default 0)
%               .t : trials
%               .i : incr
%               .w : size of initial window (default 20% of the data)
%               .g : use gain criterium (default: gain ratio)
% Be careful: it's not clear at all whether all options are really implemented
% and can be used directly. 

% addpath([IMPORT_DIR 'neuro_cvs/matlab/c4.5'])

% defaults

defaults = {'nBoots',5};

% Backward compatibility for the old signature
if nargin<3
  nBoots=5;
  opt = [];
elseif nargin == 3
  if ~isstruct(varargin{1}) && ~ischar(varargin{1})
    nBoots = varargin{1};
    opt = [];
  else
    opt =  set_properties(varargin, defaults);
    nBoots = opt.nBoots;
  end
elseif nargin == 4
  if ~ischar(varargin{1})
    nBoots = varargin{1};
    opt = varargin{2};
  else
    opt = set_properties(varargin, defaults);
    nBoots = opt.nBoots;
  end	  
else % new signature
  opt = set_properties(varargin, defaults);
  nBoots = opt.nBoots;
end


% if nargin<3, nBoots=5; end
% if nargin<4, 
%   opt= []; 
% elseif ~isstruct(opt),
%   m= opt;
%   clear opt;
%   opt.m= m;
% end


mopt = opt;
clear opt;
opt = struct([]);

fn = fieldnames(mopt);
for i=1:length(fn)
  if ~equal(fn{i},'nBoots') && ~equal(fn{i},'isPropertyStruct')
    opt.(fn{i}) = mopt.(fn{i});
   end
end


if size(yTr,1)==2,
  yTr= [-1 1]*yTr;
end

C.xTr= xTr;
C.yTr= yTr;
C.nBoots= nBoots;
C.opt= opt;
