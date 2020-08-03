function axis_legendIntoCorner(no, hleg)

ax= gca;
pos= get(ax, 'position');
if nargin<2,                 %% does not work so far
  Hleg= findobj(ax, 'Tag','legend');
  hleg= [];
  ii= 0;
  while ii<length(Hleg),
    ii= ii+1;
    ud= get(Hleg(ii), 'UserData');
    if ud.PlotHandle==ax,
      hleg= Hleg(ii);
    end
  end
  if isempty(hleg),
    error('no legend found');
  end
end
legpos= get(hleg, 'position');

switch(no),
 case {1, 'upperright','ur'},
  legpos([1 2])= pos([1 2]) + pos([3 4]) - legpos([3 4]);
 case {2, 'upperleft','ul'},
  legpos(1)= pos(1);
  legpos(2)= pos(2) + pos(4) - legpos(4);
 case {3, 'lowerleft','ll'},
  legpos([1 2])= pos([1 2]);
 case {4, 'lowerright','lr'},
  legpos(1)= pos(1) + pos(3) - legpos(3);
  legpos(2)= pos(2);
 otherwise,
  error('unknown position');
end

set(hleg, 'position',legpos);
moveObjectForth(hleg);
