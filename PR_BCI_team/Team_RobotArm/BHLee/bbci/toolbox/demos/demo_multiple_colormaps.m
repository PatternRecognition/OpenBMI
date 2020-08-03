clf;
subplot(2, 3, 1);
colormap(hot(11));
imagesc([1:100]); colorbar;

acm= fig_addColormap(copper(21));
subplot(2, 3, 2);
imagesc(100+[1:100]'); colorbar('horiz');
fig_acmAdaptCLim(acm);

acm= fig_addColormap(jet(31));
subplot(2, 3, 3);
imagesc(200+[1:100]); colorbar;
fig_acmAdaptCLim(acm);

subplot(2, 3, 4);
acm= fig_addColormap(bone(41));
imagesc([1:10:1000]'); colorbar('horiz');
fig_acmAdaptCLim(acm);

subplot(2, 3, 5);
acm= fig_addColormap(cool(51));
imagesc(-10000+[1:10:1000]); colorbar;
fig_acmAdaptCLim(acm);

subplot(2, 3, 6);
acm= fig_addColormap(pink(61));
imagesc([0:0.01:1]); colorbar('horiz');
fig_acmAdaptCLim(acm);
