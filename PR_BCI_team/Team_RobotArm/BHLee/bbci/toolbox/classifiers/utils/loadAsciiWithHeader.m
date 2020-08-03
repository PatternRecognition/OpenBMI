function [X,rowNames,colNames,extraHeader] = loadAsciiWithHeader(fname,varargin)
% loadAsciiWithHeader - Load ASCII file with row and column headers
%
% Synopsis:
%   X = loadAsciiWithHeader(fname)
%   [X,rowNames,colNames,extraHeader] = loadAsciiWithHeader(fname)
%   [X,rowNames,colNames,extraHeader] = loadAsciiWithHeader(fname,'Property',Value,...)
%   
% Arguments:
%  fname: String, name of the file to read
%   
% Returns:
%  X: [m n] data matrix read from the file. Entries that could not be
%      correctly read are marked with NaNs
%  rowNames: [m 1] cell string. The string description for each line
%      (row) in the data. With the 'rowNames' options >1, the size is [m
%      opt.rowNames]
%  colNames: [n 1] cell string with column labels, taken from the header
%      line
%  extraHeader: Cell string, content of the extra header lines at the
%      beginning of the file, before the actual data starts. This is only
%      generated if the 'extraHeader' information is specified.
%   
% Properties:
%  delimiter: Delimiter character for function textread. Default: '[ ]+'
%      (space)
%  whitespace: Whitespace character(s) for function textread. A default
%      is chosen based on the 'delimiter' option.
%  rowNames: Scalar. If >0, the first rowNames columns of each line
%      contain description strings. Default: 1
%  hasHeader: True if the file has a header line containing the
%      description for each column. Default: True
%  extraHeader: Scalar. Files can have a number of extra lines at the
%      beginning that do not carry useful information. extraHeader is the
%      number of such lines. Default: 0
%  safeRead: Logical. If true, read everything from the file as strings,
%      and convert to numeric values in a subsequent step. This is more
%      memory- and time-consuming that the default case (safeRead=false)
%      where data is read from the file numerically. Use safeRead==True if,
%      for example, the file contains 'null' to mark missing values. Default:
%      False
%   
% Description:
%   loadAsciiWithHeader is a helper function to load files that look like this:
%       City    Time   Temp
%       Dallas   12      98
%       Tulsa    13      99
%       Boise    14      97
%   In this case, the funcion would return X = [12 98; 13 99; 14 97],
%   with rowNames={'Dallas', 'Tulsa', 'Boise'} and colNames={'Time',
%   'Temp'};
%   A more complicated file:
%       Some descriptors
%       No.	NAME	MW	AMW
%       X	000501-36-0	228.26	7.87
%       X	001069-66-5	144.24	5.55
%       X	083015-26-3	255.39	6.38
%   loadAsciiWithHeader(filename, 'delimiter', '\t', 'rowNames', 2,'extraHeader', 1)
%   will return
%   X =
%          228.26         7.87
%          144.24         5.55
%          255.39         6.38
%   rowNames =
%       'X'    '000501-36-0'
%       'X'    '001069-66-5'
%       'X'    '083015-26-3'
%   colNames =
%       'MW'
%       'AMW'
%   extraHeader =
%       'Some descriptors'
%   
%   
% See also: textread,strread
% 

% Author(s), Copyright: Anton Schwaighofer, Sep 2005
% $Id: loadAsciiWithHeader.m,v 1.2 2006/02/06 14:37:55 neuro_toolbox Exp $

error(nargchk(1, inf,nargin))
opt = propertylist2struct(varargin{:});
[opt,isdefault] = set_defaults(opt, 'delimiter', '[ ]+', ...
                                    'whitespace', [], ...
                                    'rowNames', 1, ...
                                    'hasHeader', 1, ...
                                    'extraHeader', 0, ...
                                    'safeRead', 0);

error(nargchk(1, inf, nargin));

rowNames = {};
colNames = {};

% Check a few standard cases for delimiter and whitespace:
if isdefault.whitespace,
  switch opt.delimiter
    case ' ', '[ ]+'
      opt.whitespace = '';
    case '\t'
      opt.whitespace = ' \b';
    otherwise
      opt.whitespace = '';
  end
end

if opt.hasHeader,
  headerlines = 1+opt.extraHeader;
else
  headerlines = opt.extraHeader;
end

bufsize = 8191;
% Parameter parts that are relevant for strread as well
strreadParams = {'delimiter', opt.delimiter, 'whitespace', opt.whitespace, 'bufsize', ...
                bufsize};
% Params relevant to textread:
textreadParams = {'emptyvalue', NaN, 'headerlines', headerlines};

% Read the first lines of the file. That are the extra header lines, plus
% the actual column header line
firstLine = textread(fname, '%s', opt.extraHeader+1, 'delimiter', '\n', 'whitespace', '', ...
                     'bufsize', bufsize);
extraHeader = firstLine(1:opt.extraHeader);
% Split up into parts:
s = strread(firstLine{end}, '%s', strreadParams{:});
% Only the part starting after the row label is relevant:
if opt.rowNames>0,
  s = s((opt.rowNames+1):end);
end
% The effective number of data columns, determined from the first line of
% the file (header or first line of data)
nColumns = length(s);
if opt.hasHeader,
  % The first line is header: s are the column labels
  colNames = s;
end

% The first entries are the description strings for each row
formatString = repmat('%s', [1 opt.rowNames]);
% then come the actual data columns (numeric). Originally, I used %f
% here, but DRAGON files have strange 'null' entries. Thus, use strings
% and str2double
if opt.safeRead,
  formatString = [formatString repmat('%s', [1 nColumns])];
else
  formatString = [formatString repmat('%n', [1 nColumns])];
end
% Make a large cell array to hold all columns of the data
fullData = cell(1, nColumns);

% read data, writing each column into one entry of the cell array
if opt.rowNames>0,
  rowNames = cell(1, opt.rowNames);
  [rowNames{:}, fullData{:}] = textread(fname, formatString, strreadParams{:}, ...
                                        textreadParams{:});
  rowNames = horzcat(rowNames{:});
else
  fullData{:} = textread(fname, formatString, strreadParams{:}, ...
                         textreadParams{:});
end
if opt.safeRead,
  for i = 1:nColumns,
    fullData{i} = str2double(fullData{i});
  end
end
% Concatenate all data into a numeric array
X = horzcat(fullData{:});
