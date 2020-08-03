function [handle,target_rect] = feedback_2d_init(fig,fb_opt);

gray = [1,1,1]*0.8;
GRAY = [1,1,1]*0.8;
blue = [0 0 1];
green = [0 1 0];
red = [1 0 0];

clf;
set(fig, 'Menubar','none', 'Resize','off', ...
         'position',fb_opt.position);
hold on;
set(fig,'color',[1 1 1], 'DoubleBuffer','on', ...
        'Pointer','custom', 'PointerShapeCData',ones(16)*NaN);
set(get(fig,'Children'), 'position',[0 0 1 1], ...
                  'XLim',[-1 1], 'YLim',[-1 1]);
axis off

d= fb_opt.target_dist;
w= fb_opt.target_width;
switch(fb_opt.target_mode),
 case 1,
  target_rect= [0   d w 1-2*d;
                1-w d w 1-2*d];
      
 case 2,
  target_rect= [d 0   1-2*d w;
                d 1-w 1-2*d w];
 case 3,
  target_rect= [0   w+d w 1-2*d-2*w;
                1-w w+d w 1-2*d-2*w;
                w+d 0   1-2*d-2*w w;
                w+d 1-w 1-2*d-2*w w];
 case 4,
  %    ext= [(1-d)/2 (1-d)/2];
  target_rect= [0   0   w w;
                0   1-w w w;
                1-w 1-w w w;
                1-w 0   w w];
  
  
 case 5,
  l= (1-2*w-3*d)/2;
  target_rect= [0   w+d w 1-2*d-2*w;
                w+d 1-w l w;
                w+2*d+l 1-w l w;
                1-w w+d w 1-2*d-2*w;
                w+d     0 l w;
                w+2*d+l 0 l w];
 case 6,
  target_rect= [0   w+d w  1-2*d-2*w;
                0   1-w w  w;
                w+d 1-w 1-2*d-2*w w;
                1-w 1-w w  w;
                1-w w+d w 1-2*d-2*w;
                1-w 0   w  w;
                w+d 0   1-2*d-2*w w;
                0   0   w  w];
 case 7,
  l= (1-2*w-3*d)/2;
  target_rect= [0   w+d w l;
                0 w+2*d+l w l;
                w+d 1-w l w;
                w+2*d+l 1-w l w;
                1-w w+2*d+l w l;
                1-w w+d w l;
                w+d     0 l w;
                w+2*d+l 0 l w];
 case 8,   
  m = cot(pi/3)/2+0.5;
  l = 2*max(0,1-m-0.5*d);
  target_rect= [m-0.5*l, 0, l,w;
                m-0.5*l 1-w l w;
                0   0.5-0.5*l w l];
 case 9,
  m = 0.5-cot(pi/3)/2;
  l = 2*max(0,m-0.5*d);
  target_rect= [m-0.5*l 0, l w;
                m-0.5*l 1-w,l,w;
                1-w 0.5-0.5*l w l];
 
end
w1 = fb_opt.next_target_width*w;
switch(fb_opt.target_mode),
 case 1,
  target_rect2= [0   d w1 1-2*d;
                1-w1 d w1 1-2*d];
 case 2,
  target_rect2= [d 0   1-2*d w1;
                d 1-w1 1-2*d w1];
 case 3,
  target_rect2= [0   w+d w1 1-2*d-2*w;
                1-w1 w+d w1 1-2*d-2*w;
                w+d 0   1-2*d-2*w w1;
                w+d 1-w1 1-2*d-2*w w1];
 case 4,
  %    ext= [(1-d)/2 (1-d)/2];
  target_rect2= [0   0   w1 w1;
                0   1-w1 w1 w1;
                1-w1 1-w1 w1 w1;
                1-w1 0   w1 w1];
  
  
 case 5,
  l= (1-2*w-3*d)/2;
  target_rect2= [0   w+d w1 1-2*d-2*w;
                w+d 1-w1 l w1;
                w+2*d+l 1-w1 l w1;
                1-w1 w+d w1 1-2*d-2*w;
                w+d     0 l w1;
                w+2*d+l 0 l w1];
 case 6,
  target_rect2= [0   w+d w1  1-2*d-2*w;
                0   1-w1 w1  w1;
                w+d 1-w1 1-2*d-2*w w1;
                1-w1 1-w1 w1  w1;
                1-w1 w+d w1 1-2*d-2*w;
                1-w1 0   w1  w1;
                w+d 0   1-2*d-2*w w1;
                0   0   w1  w1];
 case 7,
  l= (1-2*w-3*d)/2;
  target_rect2= [0   w+d w1 l;
                0 w+2*d+l w1 l;
                w+d 1-w1 l w1;
                w+2*d+l 1-w1 l w1;
                1-w1 w+2*d+l w1 l;
                1-w1 w+d w1 l;
                w+d     0 l w1;
                w+2*d+l 0 l w1];
 case 8,
  l= (1-2*w-3*d)/2;
  target_rect2= [0   w+d w1 1-2*d-2*w;
                w+2*d+l 1-w1 l w1;
                w+2*d+l 0 l w1];
 case 9,
  l= (1-2*w-3*d)/2;
  target_rect2= [w+d 1-w1 l w1;
                1-w1 w+d w1 1-2*d-2*w;
                w+d     0 l w1];
 
end
nTargets= size(target_rect,1);
hreg= zeros(1, nTargets);
hreg2= zeros(1, nTargets);
%% scale [0 1]-rects to [-1 1]-rects
target_rect(:,[3 4])= 2*target_rect(:,[3 4]);
target_rect(:,[1 2])= 2*target_rect(:,[1 2])-ones(nTargets,2);
target_rect= target_rect(:,[1 2 1 2]) + ...
    [zeros(nTargets,2) target_rect(:, [3 4])];

target_rect2(:,[3 4])= 2*target_rect2(:,[3 4]);
target_rect2(:,[1 2])= 2*target_rect2(:,[1 2])-ones(nTargets,2);
target_rect2= target_rect2(:,[1 2 1 2]) + ...
    [zeros(nTargets,2) target_rect2(:, [3 4])];
for tt= 1:nTargets,
  hreg(tt)= patch(target_rect(tt,[1 3 3 1]), target_rect(tt,[2 2 4 4]), ...
                  gray);
  hreg2(tt)= patch(target_rect2(tt,[1 3 3 1]), target_rect2(tt,[2 2 4 4]), ...
                  gray);
end
set(hreg, 'EdgeColor','none');
set(hreg2, 'EdgeColor','none');

pat = patch([-1 1 1 -1]*fb_opt.free_region, ...
            [-1 -1 1 1]*fb_opt.free_region, GRAY);
set(pat, 'EdgeColor','none');

if fb_opt.free_region>=1,
  set(pat, 'Visible','off');
else
  set(pat, 'Visible','on');
end

cou = text(0, 0, '');

set(cou,'HorizontalAlignment','center', 'FontSize',100,'FontName','utopia');

ht= text(-0.5,0.99,'HIT: 0');
ht(2)= text(0.5,0.99,'MISS: 0');
set(ht,'VerticalAlignment','top', 'HorizontalAlignment','center', ...
       'FontSize',40,'FontName','utopia');

cross = plot(0,0,fb_opt.marker_nonactive);
set(cross, 'MarkerSize',fb_opt.marker_nonactive_size(1), ...
           'LineWidth',fb_opt.marker_nonactive_size(2), ...
           'eraseMode','xor');


if fb_opt.next_target==0
  set(hreg2,'Visible','off');
end

handle = [cross, pat, cou, ht,fig,gcf,hreg,hreg2];
