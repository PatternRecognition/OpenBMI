function fv= load_features_from_ascii(file, varargin)
%fv= load_features_from_ascii(file, <opts>)
%
% Reads ASCII files, does (if appropriate) some symbol to numeric
% transformation, separates features from labels and puts the information
% into a feature vector struct as used in the BBCI Toolbox.
%
% IN file   - file name (full path and extension)
%    opts   struct (or property/value list) with fields
%     .label_idx      - index of the column in the file that holds the
%                       labels, default 1
%     .ignore_idx     - indices of columns that should be discarded,
%                       default []
%     .delimiter_list - an array of characters that potentially separate
%                       columns in the file, default [',' ' ']
%                       (The symbols are tried in turn until one
%                       works, i.e., recognizes more than one column).
%
% OUT  fv    feature vector struct with fields
%       .x         - data matrix [nDim nSamples]
%       .y         - label affiliation matrix [nClasses nSamples]
%       .className - cell array of class names
%       .title     - file name
%
% See also cell2numeric, classmarker2mat

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'delimiter_list', [',' ' '], ...
                  'label_idx', 1, ...
                  'regression', 0, ...
                  'ignore_idx', []);

ii= 0;
nFeatures= 1;
while nFeatures==1 & ii<length(opt.delimiter_list),
  ii= ii+1;
  del= opt.delimiter_list(ii);
  [data]= textread(file, '%s', 'delimiter','\n', 'whitespace',del);
  line= strread(data{1}, '%s', 'delimiter',del);
  nFeatures= length(line);
end
if nFeatures==1,
  error('only one column recognize. check file format or opt.delimiter_list');
elseif opt.label_idx>nFeatures,
  error('label index exceeds number of columns');
end

[data]= textread(file, '%s', 'delimiter',del);
N= length(data);
nSamples= N / nFeatures;
data= reshape(data, [nFeatures nSamples])';
fv= struct('x',[], 'y',[]); %% just to have fields in this order
if opt.regression,
  fv.y= cell2numeric(data(:, opt.label_idx))';
else
  [fv.y, fv.className]= classmarker2mat(data(:, opt.label_idx));
end
data= data(:, setdiff(1:nFeatures, [opt.label_idx opt.ignore_idx]))';
fv.x= cell2numeric(data);
fv.title= file;
