results_dir= [DATA_DIR 'results/siamac/'];
%results_dir= ['/mnt/usb/data/results/siamac/';

N= 70;
hit1= NaN*zeros(N, 1);
hit2= NaN*zeros(N, 1);

for k= 1:N,
  file= sprintf('%s/y_mb_e30/y_imag_%03d',results_dir, k);
  if ~exist([file '.mat'], 'file'),
    continue;
  end
  load(sprintf('%s/imagy_%03d',results_dir, k));  %% epoy
%  S= load(file); 
%  y1= S.y;  
  S= load(file);
  y2= S.y;  
%  hit1(k)= 100*mean([1 2]*epoy==[1 2]*y1);
  hit2(k)= 100*mean([1 2]*epoy==[1 2]*y2);
end
