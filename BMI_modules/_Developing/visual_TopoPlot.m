function visual_TopoPlot(SMT, w, CNT)

MNT = opt_getMontage(SMT);

center = [0 0];                   
theta = linspace(0,2*pi,360);  

x = cos(theta)+center(1);  
y = sin(theta)+center(2);  

oldUnit = get(gcf,'units');
set(gcf,'units','normalized');

H = struct('ax', gca);
set(gcf,'CurrentAxes',H.ax);

% ----------------------------------------------------------------------
% nose plot
nose = [1 1.2 1];
nosi = [83 90 97]+1;
H.nose = plot(nose.*x(nosi), nose.*y(nosi), 'k', 'linewidth', 2.4); 
hold on;

% ----------------------------------------------------------------------
% ears plot
earw = .08; earh = .3;
H.ears(1) = plot(x*earw-1-earw, y*earh, 'k',  'linewidth', 2);
H.ears(2) = plot(x*earw+1+earw, y*earh, 'k',  'linewidth', 2);
hold on;

% ----------------------------------------------------------------------
% main circle plot
H.main = plot(x,y, 'k', 'linewidth', 2.2);                 
set(H.ax, 'xTick',[], 'yTick',[]);
axis('xy', 'tight', 'equal', 'tight');
hold on;

% ----------------------------------------------------------------------
% Rendering the contourf
xe_org = MNT.x';
ye_org = MNT.y';

w = w(:);
resolution = 101;

maxrad = max(1,max(max(abs(MNT.x)),max(abs(MNT.y))));

xx = linspace(-maxrad, maxrad, resolution);
yy = linspace(-maxrad, maxrad, resolution)';

xe_add = cos(linspace(0,2*pi,resolution))'*maxrad;
ye_add = sin(linspace(0,2*pi,resolution))'*maxrad;
w_add = ones(length(xe_add),1)*mean(w);

xe = [xe_org; xe_add];
ye = [ye_org; ye_add];
w = [w; w_add];

xe_add = cos(linspace(0,2*pi,resolution))';
ye_add = sin(linspace(0,2*pi,resolution))';

xe = [xe;xe_add];
ye = [ye;ye_add];
w = [w; zeros(length(xe_add),1)];

[xg,yg,zg] = griddata(xe, ye, w, xx, yy);
contourf(xg, yg, zg, 50, 'LineStyle','none'); hold on;

% ----------------------------------------------------------------------
% disp electrodes
for i = 1:size(xe_org,1)
    plot(xe_org(i), ye_org(i), 'k*'); hold on;
end

axis off;
colorbar('vert');

end

