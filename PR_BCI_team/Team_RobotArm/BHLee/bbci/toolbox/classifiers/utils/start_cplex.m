global LPENV
if isempty(LPENV) | (LPENV==0),
  [LPENV,OK]= cplex_init(1);
  waiting_time = 0;
  while (OK~=0 & waiting_time < 3600)
    warning(['No CPLEX license is available. Trying again after one' ...
	     ' minute...']);
    pause(60);
    waiting_time = waiting_time + 60;
    [LPENV,OK]= cplex_init(1);
  end
  if (OK ~= 0)
    error(['No CPLEX license available withing an hour. Bailing' ...
	   ' out...']);
  end
end
