% readbvconf() - read Brain Vision Data Exchange format configuration 
%                file
%
% Usage:
%   >> CONF = readbvconf(pathname, filename);
%
% Inputs:
%   pathname  - path to file
%   filename  - filename
%
% Outputs:
%   CONF      - structure configuration
%
% Author: Andreas Widmann, University of Leipzig, 2007

% Copyright (C) 2007 Andreas Widmann, University of Leipzig, widmann@uni-leipzig.de
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

% $Id: readbvconf.m 44 2009-11-12 02:00:56Z arnodelorme $

function CONF = readbvconf(filename)

% if nargin < 2  changed by mhlee 2015_09_17
%     error('Not enough input arguments');
% end
% Open and read file
[IN, message] = fopen(fullfile(filename));
if IN == -1
    [IN, message] = fopen(fullfile(lower(filename)));
    if IN == -1
        error(message)
    end;
end
raw={};
while ~feof(IN)
    raw = [raw; {fgetl(IN)}];
end
fclose(IN);

% Remove comments and empty lines
raw(strmatch(';', raw)) = [];
raw(cellfun('isempty', raw) == true) = [];

% Find sections
sectionArray = [strmatch('[', raw)' length(raw) + 1];
for iSection = 1:length(sectionArray) - 1

    % Convert section name
    fieldName = lower(char(strread(raw{sectionArray(iSection)}, '[%s', 'delimiter', ']')));
    fieldName(isspace(fieldName) == true) = [];

    % Fill structure with parameter value pairs
    switch fieldName
        case {'commoninfos' 'binaryinfos'}
            for line = sectionArray(iSection) + 1:sectionArray(iSection + 1) - 1
                splitArray = strfind(raw{line}, '=');
                CONF.(fieldName).(lower(raw{line}(1:splitArray(1) - 1))) = raw{line}(splitArray(1) + 1:end);
            end
        case {'channelinfos' 'coordinates'}
            for line = sectionArray(iSection) + 1:sectionArray(iSection + 1) - 1
                splitArray = strfind(raw{line}, '=');
                tmp = str2num(strtrim(raw{line}(3:splitArray(1) - 1)));
                if isempty(tmp)
                    tmp = NaN; end                
                CONF.(fieldName)(tmp) = {raw{line}(splitArray(1) + 1:end)};
            end
        case {'markerinfos'} % Allow discontinuity for markers (but not channelinfos and coordinates!)
            for line = sectionArray(iSection) + 1:sectionArray(iSection + 1) - 1
                splitArray = strfind(raw{line}, '=');
                tmp = str2num(strtrim(raw{line}(3:splitArray(1) - 1)));                
                if isempty(tmp)
                    tmp = NaN; end
                CONF.(fieldName)(line - sectionArray(iSection), :) = {raw{line}(splitArray(1) + 1:end) tmp};
            end
            if ~all(1:size(CONF.(fieldName), 1) == [CONF.(fieldName){:, 2}])
                warning('Marker number discontinuity.')
            end
        case 'comment'
            CONF.(fieldName) = raw(sectionArray(iSection) + 1:sectionArray(iSection + 1) - 1);
    end
end
