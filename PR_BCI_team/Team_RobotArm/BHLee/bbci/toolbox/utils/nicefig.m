function nicefig(ip, figName)
%nicefig(<ip, figName>)

global DISPLAY_SIZE DISPLAY_OFFSET

hf= figure;
if ~exist('ip','var'), ip=gcf; end
if ~exist('figName','var'), [d,figName]= system('hostname'); end

if isempty(DISPLAY_SIZE),
%  set(hf, 'menuBar','none', 'toolbar','none', 'position', [0 0 10000 10000]);
  set(hf, 'menuBar','none', 'toolbar','none', 'position', [0 0 1280 1024]);
  drawnow;
  pp= get(hf, 'position');
  DISPLAY_SIZE= pp(3:4);
  DISPLAY_OFFSET= pp(1:2);
  set(hf, 'menuBar','figure');
end

bb= [DISPLAY_OFFSET([1:2]), 55, 27];
w= floor(DISPLAY_SIZE(1)/2);
h= floor(DISPLAY_SIZE(2)/2) - bb(4);
ip= mod(ip-1,4)+1;
switch(ip),
 case 1, pos= [bb(1), DISPLAY_SIZE(2)+bb(2)-bb(4)-h];
 case 2, pos= [DISPLAY_SIZE(1)-w+bb(1) DISPLAY_SIZE(2)+bb(2)-bb(4)-h];
 case 3, pos= [bb(1), DISPLAY_SIZE(2)+bb(2)-bb(3)-bb(4)-h-h];
 case 4, pos= [DISPLAY_SIZE(1)-w+bb(1) DISPLAY_SIZE(2)+bb(2)-bb(3)-bb(4)-h-h];
end

set(hf, 'position',[pos w h], 'numberTitle','off', ...
         'numberTitle','off', 'toolbar','none', ...
         'name',sprintf('fig@%s.%d', figName, hf));
