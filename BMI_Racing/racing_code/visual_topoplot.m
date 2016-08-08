function visual_topoplot(w, xe, ye, xx, yy, scale)

% w = w(:);
% 
% w_add = ones(length(xe_add),1)*mean(w);
% 
% w = [w; w_add];
% w = [w; zeros(length(xe_add),1)];
[xg,yg,zg] = griddata(xe, ye, w, xx, yy);

surf(xg,yg,zg);
view(2);
caxis(scale);

axis off;
colorbar('vert');

end

