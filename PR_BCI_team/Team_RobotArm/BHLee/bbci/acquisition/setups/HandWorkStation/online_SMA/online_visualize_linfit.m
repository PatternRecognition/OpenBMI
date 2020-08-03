
function wld = online_visualize_linfit(wld)


figure(wld.fig.h)
induced_offset = 3;


% plot detector state
subplot(1,6,1), hold off
bar(0,0), hold on
if wld.state.n_lo>0
    if wld.state.n_lo<wld.wlen-1
        bar(0,-wld.state.n_lo,'facecolor',[.5 .5 .5],'barwidth',1)
    else
        bar(0,-wld.state.n_lo,'g','barwidth',1)
    end
elseif wld.state.n_hi>0
    if wld.state.n_hi<wld.wlen-1
        bar(0,wld.state.n_hi,'facecolor',[.5 .5 .5],'barwidth',1)
    else
        bar(0,wld.state.n_hi,'r','barwidth',1)
    end
else
    bar(0,0,'barwidth',1)
end
box on
set(gca,'xlim',[-.5 .5],'ylim',[-(wld.wlen-1) wld.wlen-1],'xtick',[],'ytick',[])


% plot detector progress
subplot(1,6,2:6), hold on
if wld.ni>=2
    if wld.induced==-1
        patch([wld.ni-1 wld.ni wld.ni wld.ni-1]*wld.dt,[wld.fig.ylim(2)-induced_offset wld.fig.ylim(2)-induced_offset wld.fig.ylim(2) wld.fig.ylim(2)],[.9 1 .9],'edgecolor',[.9 1 .9],'linewidth',.001)
        patch([wld.ni-1 wld.ni wld.ni wld.ni-1]*wld.dt,[wld.fig.ylim(1) wld.fig.ylim(1) wld.fig.ylim(1)+induced_offset wld.fig.ylim(1)+induced_offset],[.9 1 .9],'edgecolor',[.9 1 .9],'linewidth',.001)
    elseif wld.induced==1
        patch([wld.ni-1 wld.ni wld.ni wld.ni-1]*wld.dt,[wld.fig.ylim(2)-induced_offset wld.fig.ylim(2)-induced_offset wld.fig.ylim(2) wld.fig.ylim(2)],[1 .9 .9],'edgecolor',[1 .9 .9],'linewidth',.001)
        patch([wld.ni-1 wld.ni wld.ni wld.ni-1]*wld.dt,[wld.fig.ylim(1) wld.fig.ylim(1) wld.fig.ylim(1)+induced_offset wld.fig.ylim(1)+induced_offset],[1 .9 .9],'edgecolor',[1 .9 .9],'linewidth',.001)
    end
    plot([wld.ni-1:wld.ni]*wld.dt,wld.y(wld.ni-1:wld.ni),'color',[.25 .25 .25],'linewidth',1)
end
if wld.ni>=wld.wlen+1
    plot([wld.ni-1:wld.ni]*wld.dt,wld.y_lp(wld.ni-wld.wlen:wld.ni-wld.wlen+1),'color',[.75 .75 .75],'linewidth',2)
end

if wld.state.hi
    %plot([wld.ni-wld.wlen+1:wld.ni]*wld.dt,wld.y_lp(wld.ni-wld.wlen*2+2:wld.ni-wld.wlen+1),'r','linewidth',3)
    plot([wld.ni wld.ni]*wld.dt,wld.fig.ylim+[induced_offset -induced_offset],'r','linewidth',3)
end
if wld.state.lo
    %plot([wld.ni-wld.wlen+1:wld.ni]*wld.dt,wld.y_lp(wld.ni-wld.wlen*2+2:wld.ni-wld.wlen+1),'g','linewidth',3)
    plot([wld.ni wld.ni]*wld.dt,wld.fig.ylim+[induced_offset -induced_offset],'g','linewidth',3)
end

set(gca,'xlim',[0 wld.T],'ylim',wld.fig.ylim,'ytick',[])
box on







