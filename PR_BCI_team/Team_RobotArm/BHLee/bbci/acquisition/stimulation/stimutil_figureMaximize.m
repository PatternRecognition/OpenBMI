function oldPos= stimutil_figureMaximize(pos)
%STIMUTIL_FIGUREMAXIMIZE - Maximize Matlab Figure to a Specified Position
%
%Synopsis:
% stimutil_figureMaximize(POSITION)
%
%Arguments:
% POSITION: 

% OPT: struct or property/value list of optional arguments:

% blanker@cs.tu-berlin.de, Jul-2007


%opt= propertylist2struct(varargin{:});
%opt= set_defaults(opt, ...


oldPos= get(gcf, 'position');

if ~ischar(pos) & length(pos)~=4,
  switch(pos),
   case 0,
    pos= oldPos;
   case {1,'central'}
    %  pos= [-1280 176 1280 1024];
    pos= [0 0 1600 1200];
    set(gcf, 'pointer', 'custom', 'pointerShapeCData', repmat(NaN, 16, 16));
   case {2, 'right'},
    %   pos= [-1279 1 1280 1005];
    %   pos= [1281 1 1280 1005];
    pos= [1282 0 1280 1024];
   case {2.5, 'left'},
    pos= [-1280 1 1280 1024];
   case {3, 'below'},
    pos= [1 -600 800 601];
    %pos= [1 -1025 1280 1025];
   case 'belowhighres',
    pos= [1 -1024 1280 1025];
   case {4, 'small'},
    pos= oldPos;
   case {6, 'laptop'},
    set(gcf,'menuBar','none', 'numberTitle','off');
    pos= [1 1 1400 1050];
   case {5, 'extern'},
    set(gcf,'menuBar','none', 'numberTitle','off');
    pos= [1401 1 1280 1050];
   case {7,'vd2'},
    set(gcf,'menuBar','none', 'numberTitle','off');
    pos= [1 -1024 1280 1025];
   case {8,'smallmoni'},
    set(gcf,'menuBar','none', 'numberTitle','off');
    pos= [1 -768 1024 769];
   case {9,'verysmallmoni'},
    set(gcf,'menuBar','none', 'numberTitle','off');
    pos= [1 -478 616 478];
  end
end  

oldUnits= get(gcf, 'units');
set(gcf, 'color', 0.5*[1 1 1], 'DoubleBuffer', 'on');
set(gcf, 'menuBar','none', 'units','pixel', ...
         'position',pos, 'units', oldUnits);
set(gcf, 'DefaultAxesPosition', [0 0 1 1]);
set(gcf, 'NumberTitle', 'off');
clf;
figure(gcf);
