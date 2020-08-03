function timeLimit= getTimeLimit(paceStr)
%timeLimit= getTimeLimit(paceStr)

%% delete appendices like '_fb' ind '1_5s_fb'
ii= findstr('s', paceStr);
if ii<length(paceStr) & paceStr(ii+1)=='_',
  paceStr= paceStr(1:ii);
end

switch(paceStr),
 case '5s', timeLimit= 8000;
 case '2s', timeLimit= 4000;
 case '1_5s', timeLimit= 3000;
 case '1s', timeLimit= 2000;
 case {'0_5s','0_3s'}, timeLimit= 1000;
 otherwise, 
  timeLimit= 4000;
  warning('pace not known');
end
