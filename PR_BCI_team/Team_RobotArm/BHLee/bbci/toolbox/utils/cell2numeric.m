function data= cell2numeric(data_sym, varargin)
%data= cell2numeric(data_sym, <opt>)
%
% Transform data matrices as read from ASCII files to numeric matrices.
%
% Transforms a data matrix given as an character cell array into a 
% numeric matrix. Each row may (1) either contain continuous values (doubles)
% in character format (these are converted by str2double), or (2) contain
% strings from a (typically small) set of attributes, e.g., 
% {'low', 'medium', 'high'}. For each row the set of attributes is
% determined, brought into alphabetical sequence, and for conversion to
% numerical values each attribut is mapped to the rank in that order.
%
% The 'argument' opt may be a stuct or a property list:
% .symbol_for_unknown - a string or a cell array of strings specifying
%                       what symbol(s) are used to mark unknown values
%                       in symbolic rows, default {'?', '-'}. 
%                       NOTE: in numeric rows unknown values must be
%                       maked 'NaN' otherwise the whole row is interpreted
%                       to be symbolic (see 'funny effect' below).
% .format - with this field one can specify the format of the
%   cell array as 'pure_symbolic', 'pure_numeric', 'mixed'. The default
%   is 'mixed'. The advantage of using 'pure_symbolic' or
%   'pure_numeric' is that it is faster in those cases. Using
%   'pure_numeric' for mixed cell arrays leads to NaN entries for 
%   non-numeric elements, and using 'pure_symbolic' for mixed cell arrays
%   leads to funny effects ({'2','6','4'} -> [1 3 2] as conversion is
%   based on the abstract attribute sequence <'2','4','6'>).
%
% Example
%  C= {'4', '3.1', '1', '5.5', 'NaN', '7';
%      'low', 'high', 'low', 'medium', 'high', '?'};
%  cell2numeric(C)
% ans =
%      4   3.1     1   5.5   NaN     7
%      2     1     2     3     1   NaN

% bb ida.first.fhg.de


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'format', 'mixed', ...
                  'symbol_for_unknown', {'?','-'});

if ~iscell(opt.symbol_for_unknown),
  opt.symbol_for_unknown= {opt.symbol_for_unknown};
end

[nDim, nSamples]= size(data_sym);
switch(opt.format),
 case 'pure_symbolic',
  data= zeros(nDim, nSamples);
  idx_sym= 1:nDim;
 case 'pure_numeric',
  data= str2double(data:sym);
  return;
 case 'mixed',
  data= str2double(data_sym);
  idx_sym= find(all(isnan(data),2));
 otherwise,
  error('unknown format');
end

for nn= idx_sym,
  z= data_sym(nn,:);
  sym_set= unique(z);
  for kk= 1:length(sym_set),
    idx= strmatch(sym_set{kk}, z, 'exact');
    data(nn, idx)= kk;
  end
  for kk= 1:length(opt.symbol_for_unknown),
    idx= strmatch(opt.symbol_for_unknown{kk}, z, 'exact');
    data(nn, idx)= NaN;
  end
end
