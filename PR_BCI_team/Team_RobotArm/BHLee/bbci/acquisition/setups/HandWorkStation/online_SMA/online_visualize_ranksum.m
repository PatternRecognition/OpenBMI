
function wld = online_visualize_ranksum(wld)


figure(wld.fig.h)


%% plot current detector state
axes(wld.fig.ax_state), hold off
bar(0,0), hold on
if wld.state.n_lo > 0
    if wld.state.n_lo < -log(wld.alpha)
        bar(0,-wld.state.n_lo,'facecolor',[.5 .5 .5],'edgecolor',[.5 .5 .5],'barwidth',1)
    else
        bar(0,-wld.state.n_lo,'g','edgecolor','g','barwidth',1)
    end
elseif wld.state.n_hi > 0
    if wld.state.n_hi < -log(wld.alpha)
        bar(0,wld.state.n_hi,'facecolor',[.5 .5 .5],'edgecolor',[.5 .5 .5],'barwidth',1)
    else
        bar(0,wld.state.n_hi,'r','edgecolor','r','barwidth',1)
    end
else
    bar(0,0,'barwidth',1)
end
set(wld.fig.ax_state,'xlim',[-.5 .5],'ylim',[-(-log(wld.alpha)) -log(wld.alpha)],'xtick',[],'ytick',[])
set(get(wld.fig.ax_state,'ylabel'),'string','detector state')


%% plot classifier output
if wld.ni>=2
    axes(wld.fig.ax_Cout), hold on
    plot([wld.ni-1:wld.ni]*wld.dt,wld.y(wld.ni-1:wld.ni),'color',[0 0 0],'linewidth',1)
    set(wld.fig.ax_Cout,'ylim',[min(wld.y(:)) max(wld.y(:))])
end


%% plot induced/forced states
axes(wld.fig.ax_induced), hold on
if wld.ni>=2
    if wld.fig.induced == -1
        patch([wld.ni-1 wld.ni wld.ni wld.ni-1]*wld.dt,[0 0 1 1],[.75 1 .75],'edgecolor',[.75 1 .75],'linewidth',.001)
    elseif wld.fig.induced == 1
        patch([wld.ni-1 wld.ni wld.ni wld.ni-1]*wld.dt,[0 0 1 1],[1 .75 .75],'edgecolor',[1 .75 .75],'linewidth',.001)
    elseif wld.fig.forced == -1
        wld.fig.forced = 0;
        plot([wld.ni wld.ni]*wld.dt,[0 1],'b','linewidth',3)
    elseif wld.fig.forced == 1
        wld.fig.forced = 0;
        plot([wld.ni wld.ni]*wld.dt,[0 1],'m','linewidth',3)
    end
end


%% plot detected states
axes(wld.fig.ax_detected), hold on
if wld.state.hi
    plot([wld.ni wld.ni]*wld.dt,[0 1],'r','linewidth',3)
elseif wld.state.lo
    plot([wld.ni wld.ni]*wld.dt,[0 1],'g','linewidth',3)
end


%% plot speed progress
if wld.ni>=2
    axes(wld.fig.ax_speed), hold on
    plot([wld.ni-1:wld.ni]*wld.dt,wld.speed.history(wld.ni-1:wld.ni),'color',[0 0 0],'linewidth',1)
end




