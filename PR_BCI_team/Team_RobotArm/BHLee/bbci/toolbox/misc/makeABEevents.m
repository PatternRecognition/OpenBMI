function mrk_abe= makeABEevents(mrk, classDef)
% MAKEABEEVENTS convert EEG marker data using class Definitions.
%
% USAGE: mrk_abe= makeABEevents(mrk, classDef)
%
% IN:  mrk         struct containing markers (pos, toe, fs)
%      classDef    array containing class Definitions (class name, sync marker, boolean classdef, target marker)
% OUT: mrk_abe     struct containing class markers (pos, toe, fs, y, className)
%
% EXAMPLE:
% mrk =
%    pos: [272, 3191, 3216, 3292, 3317, 3392, 3416, 3492, 3518, 3591]
%    toe: [25,  252,  12,   1,    12,   1,    11,   1,    11,   1]
%     fs: 100
%
% classDef =
%    'S1'    'S10(-1000,0) and R74(-200,200)'            [74]    'right click'
%    'S1'    'S10(-1000,0) and ( not R74(-500,500) )'    [10]    'left no-click'
%    'S1'    'S11(-1000,0)'                              [11]    'rest'

% 21.8.2003 kraulem
[m,n]       = size(classDef);
mrk_abe.pos = [];
mrk_abe.toe = [];
mrk_abe.fs  = mrk.fs;
mrk_abe.y   = [];
mrk_abe.className = classDef(:,4)';

for ind1 = 1:m
  syncmarker  = (-1)^(classDef{ind1,1}(1)=='R')*eval(classDef{ind1,1}(2:end));
  syncarr     = find(mrk.toe == syncmarker);
  classindarr = [];

  [ABEstring, temp] = getABEstring(classDef{ind1,2});
  for ind2 = 1:length(syncarr)
    if(eval(ABEstring))
      classindarr = [classindarr,syncarr(ind2)];
    end
  end
  mrk_abe.pos = [mrk_abe.pos, mrk.pos(classindarr)];
  mrk_abe.toe = [mrk_abe.toe, ones(1,length(classindarr))*classDef{ind1,3}];
end
[mrk_abe.pos, sortind] = sort(mrk_abe.pos);
mrk_abe.toe = mrk_abe.toe(sortind);
mrk_abe.y   = zeros(m,length(mrk_abe.pos));
for ind = 1:m
  mrk_abe.y(ind,:) = (mrk_abe.toe == 1*classDef{ind,3});
end
return

function [str, rest] = getABEstring(inputstr)
% change the input string into evaluable format
str   = [];
rest  = [];

while ~isempty(inputstr)
  %cut off leading spaces 
  while inputstr(1)==' '
    inputstr = inputstr(2:end);
  end 
  [tok, inputstr] = strtok(inputstr);
  switch tok(1)
   case {'S','R'}
    % an interval is given
    [mark,interval]=strtok(tok,'(');
    stimmarker = (-1)^(mark(1)=='R')*eval(mark(2:end));
    [leftdist,rightdist]=strtok(interval,',');
    str = [str, sprintf('~isempty(find((mrk.toe == %d ) & (mrk.pos >= mrk.pos(syncarr(ind2))+(mrk.fs*(%s)/1000)) & (mrk.pos <= mrk.pos(syncarr(ind2))+(mrk.fs*(%s)/1000))))',stimmarker, leftdist(2:end), rightdist(2:end-1))];
   case 'a'
    str = [str, '&'];% and
   case 'o'
    str = [str, '|'];% or
   case 'n'
    str = [str, '~'];% not
   case '('
    [exp, inputstr] = getABEstring([tok(2:end), inputstr]);
    str = [str, '(', exp, ')'];
   case ')'
    rest = inputstr(2:end);
    return;
   otherwise
    disp('Boolean expression is not well-formed!');
  end
end
return
