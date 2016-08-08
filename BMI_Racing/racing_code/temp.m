

bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
orig_Dat=[];
band=[11 17]

buffer_size=5000;
data_size=1500;
orig_Dat=zeros(buffer_size, size(state.chan_sel,2));

%% Real-time visualization
%% Setting for real-time topoplot
load 'SMTFORRACING.mat'
MNT = opt_getMontage(SMT);
% center = [0 0];
% theta = linspace(0,2*pi,360);
% x = cos(theta)+center(1);
% y = sin(theta)+center(2);
% oldUnit = get(gcf,'units');
% set(gcf,'units','normalized');
% H = struct('ax', gca);
% set(gcf,'CurrentAxes',H.ax);
tic
xe_org = MNT.x';
ye_org = MNT.y';
resolution = 31;
maxrad = max(1,max(max(abs(MNT.x)),max(abs(MNT.y))));
xx = linspace(-maxrad, maxrad, resolution);
yy = linspace(-maxrad, maxrad, resolution)';

% % nose plot
% tic
% nose = [1 1.2 1];
% nosi = [83 90 97]+1;
% H.nose = plot(nose.*x(nosi), nose.*y(nosi), 'k', 'linewidth', 2.4);
% hold on;
%
% % ----------------------------------------------------------------------
% % ears plot
% earw = .08; earh = .3;
% H.ears(1) = plot(x*earw-1-earw, y*earh, 'k',  'linewidth', 2);
% H.ears(2) = plot(x*earw+1+earw, y*earh, 'k',  'linewidth', 2);
% hold on;
% % ----------------------------------------------------------------------
% % main circle plot
% H.main = plot(x,y, 'k', 'linewidth', 2.2);
% set(H.ax, 'xTick',[], 'yTick',[]);
% axis('xy', 'tight', 'equal', 'tight');
% hold on;
figure
while true
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    orig_Dat=[orig_Dat; data];
    if length(orig_Dat)>buffer_size % prevent overflow
        tic
        Dat=orig_Dat(end-buffer_size+1:end,:);
        orig_Dat=Dat;  %%
        Dat2.x=Dat;
        Dat2.fs=state.fs;
%         Dat=prep_resample(Dat2,500);
        Dat=Dat2.x;
        fDat=prep_filter(Dat, {'frequency', band;'fs',1000});%state.fs });
        tm.x=fDat;tm.fs=1000;
        tm2 = prep_envelope(tm);
        vDat=mean(tm2(end-1000:end, :));
        vDat=vDat*-1;
        visual_topoplot(vDat, xe_org, ye_org, xx, yy);
        drawnow;
        toc
%         pause(0.05);        
    end
end