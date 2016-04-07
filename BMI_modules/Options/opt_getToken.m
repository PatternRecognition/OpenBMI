function tokens = opt_getToken(string, delimiter)
% GET_TOKENS	To extract all tokens from a string even if a token is null.
%
% Description:	GET_TOKENS extract all tokens from a string even if one or
%		more tokens are null.  It is a replacement to iteratively calling the
%		builtin function 'strtok.m' because it did not output null tokens as
%		would be normally expected.  Note that if there are two adjacent
%		delimiters then an empty token would be output representing the token
%		that could have been present between the delimiters.
%
% Usage:	tokens = get_tokens(string, delimiter)
%
% Input:
%  string: a string from which to extract tokens.
%  delimiter:	a string containing one or more delimiters. The default
%    delimiters are whitespace characters (i.e. the ASCII codes for TAB,
%    LF, VT, FF, CR and space). Note that each delimiter is assumed to be
%    a single character; so if the delimiter string has three characters,
%    each of the three characters is considered a delimiter.
%
% Output:
%		tokens: An cell array of strings; the tokens retrieved from the input
%			character array.
%
% Example:  
%
%		>> s = 'what a great , day';
%	
%  Note that the variable 's' is a string that uses two delimiters, a space
%  and a comma.
%
%		>> get_tokens(s, ' ,')
%
%		ans =
%
%       'what'
%       'a'
%       'great'
%       ''
%       ''
%       'day'
% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright 2007  Jeff Jackson (Ocean Sciences, DFO Canada)
%   Creation Date: Oct. 20, 2006
%   Last Updated:  Jan. 22, 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Check the input arguments before continuing.

% Check to see if no input arguments were supplied.  If this is the case,
% stop execution and output an error message to the user.
if nargin == 0
  error('MATLAB:get_tokens:NrInputArguments', 'No input arguments were supplied. At least one is expected.');
% Check to see if the only input argument is a cell array.  If it isn't
% then stop execution and output an error message to the user. Also set the
% DIM value since it was not supplied.
elseif nargin == 1
	% Check to see if the input is a string.  Output a message if the input
	% value is a not a character array, then exit the function.
	if ~ischar(string)
		error('MATLAB:get_tokens:InvalidInputArgument', 'The input string must be a valid character array.');
	end
	% Set the default delimiter value to whitespace characters.
	delimiter = [9:13 32];
elseif nargin == 2
	% Check to see if the two input arguments are character strings.  If
	% either check fails then stop execution and output an error message to
	% the user.
	if ~ischar(string)
		error('MATLAB:get_tokens:InvalidInputArgument', 'The first input variable must be a valid character array.');
	end
	if ~ischar(delimiter)
		error('MATLAB:get_tokens:InvalidInputArgument', 'The second input variable must be a valid character array.');
	end
elseif nargin > 2
	% Check to see if too many arguments were input.  If there were then exit
	% the function issuing a error message to the user.
	error('MATLAB:get_tokens:TooManyInputArguments', 'Too many input arguments were supplied.  The maximum permitted is two.');
end

%% If the input arguments are valid continue processing.

% Get the sizes of the input parameters.
string_size = size(string);
delim_size = size(delimiter);

% Check to see if the character array is vertical instead of horizontal. If
% it is then change it. Do the opposite for the delimiter array.
if string_size(1) > 1
	s = string';
else
	s = string;
end
if delim_size(1) > 1
	d = delimiter;
else
	d = delimiter';
end

% Get the array sizes.
if isempty(d)
   error('MATLAB:get_tokens:InputArguments', 'Delimiter is invalid.');
end

% Surround the delimiter string with square brackets so the regexprep
% function works correctly.
delimiter2 = ['[' delimiter ']'];

% Change all delimiters to the character value of the number 1.
s2 = regexprep(s,delimiter2,char(1));

% Extract all tokens from this updated string.
tokens = strread(s2,'%s','delimiter',char(1));

% Check to see if the last character in the input string is a delimiter. If
% it was then add a cell with an empty string to the end of the tokens cell
% array.
if strcmp(s2(end), char(1)) == 1
	tokens(end+1,1) = {''};
end