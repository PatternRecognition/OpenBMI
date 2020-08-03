params = {0.3};
protocol = {3;4;4};
port = 12489;
fb = 'animate_fb_pseudo_brainrace';

global run

run = 1


[dum,rech] = system('hostname;');
disp(rech)

rech = rech(1:end-1);
if ~exist('port','var') | isempty(port)
  get_udp(rech,protocol);
else
  get_udp(rech,protocol,port);
end

feval(fb,'init',params{:});

while run
  data = get_udp;
  feval(fb,data);
end
  

get_udp('close');


  

