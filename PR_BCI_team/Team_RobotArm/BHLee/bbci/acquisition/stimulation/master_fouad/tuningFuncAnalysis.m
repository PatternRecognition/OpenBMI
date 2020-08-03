% Function that analysis the raw data and plots tuningfunction

function tuningFuncAnalysis(cnt,mrk,Para,plot_chan,plot_classes,fft_band,file,buffer,filt_on,stop_ref,start_ref,start,stop)

ind_1 = chanind(cnt,plot_chan{1});
ind_2 = chanind(cnt,plot_chan{2});

% Plots FFT of raw channels
cnt_spec= proc_spectrum(cnt, fft_band, kaiser(cnt.fs,2));

figure('Name','FFT - spectra')
imagesc(cnt_spec.t,[1:size(cnt.x,2)],cnt_spec.x');colormap('Hot');colorbar
ylabel('Number of channels')
xlabel('Frequency [Hz]')
title(['Filename: ',file, '  [',num2str(round(min(cnt_spec.x(:)))) ,' ', num2str(round(max(cnt_spec.x(:)))),']', ' dB'])

figure('Name',['FFT of channel ', plot_chan{1}, plot_chan{2}])
subplot(2,1,1)
plot(cnt_spec.t,cnt_spec.x(:,ind_1),'LineWidth',2)
xlabel('Frequency [Hz]')
ylabel('Amplitude [dB]')
grid on
title(['Channel',plot_chan{1}])

subplot(2,1,2)
plot(cnt_spec.t,cnt_spec.x(:,ind_2),'LineWidth',2)
xlabel('Frequency [Hz]')
ylabel('Amplitude [dB]')
grid on
title(['Channel',plot_chan{2}])

refDef  = {'S102';'REF'};
mrk_ref = mrk_defineClasses(mrk, refDef);
epo_ref = cntToEpo(cnt, mrk_ref,[start_ref stop_ref]);
epo_spec_ref = proc_spectrum(epo_ref, fft_band, kaiser(cnt.fs,2));
figure('Name','REF')
    subplot(2,2,1)
    plotChannel(epo_ref,plot_chan{1})
    subplot(2,2,2)
    plotChannel(epo_ref,plot_chan{2})
        subplot(2,2,3)
    plotChannel(epo_spec_ref,plot_chan{1})
    subplot(2,2,4)
    plotChannel(epo_spec_ref,plot_chan{2})

% Epoch continous data 

for i = 1:length(plot_classes),
    
    stimDef = {['S ',int2str(plot_classes(i))];['Mod-freq ',int2str(plot_classes(i))]};

    mrk_stim= mrk_defineClasses(mrk, stimDef);

    epo = cntToEpo(cnt, mrk_stim, [start stop]);
    epo_sub = proc_subtractReferenceClass(epo, epo_ref);
        if filt_on ==1
            n=5;
            r =20;
            Wn = [plot_classes(i)-1 plot_classes(i)+1]/cnt.fs*2;
            [b,a] = cheby2(n,r,Wn);
            cnt_flt = proc_filt(cnt,b,a);
            epo_flt = cntToEpo(cnt_flt, mrk_stim, [start stop]);
          end
        
    mnt= getElectrodePositions(cnt.clab);
    mnt= mnt_setGrid(mnt, 'medium');
    opt_spec= defopt_spec;
    mnt= mnt_shrinkNonEEGchans(mnt);

    figure('Name','Time-domain epochs all cannels')
    grid_plot(epo,mnt)

    figure('Name',['Time-domain epochs ', 'Mod.freq = ',int2str(plot_classes(i)), ' Hz'])
    subplot(2,2,1)
    plotChannel(epo,plot_chan{1})
    subplot(2,2,3)
    plotChannel(epo,plot_chan{2})
    subplot(2,2,2)
    plotChannel(epo_sub,plot_chan{1})
    subplot(2,2,4)
    plotChannel(epo_sub,plot_chan{2})
% Spectra of Epochs

epo_spec= proc_spectrum(epo, fft_band, kaiser(cnt.fs,2));
epo_spec_sub = proc_spectrum(epo_sub, fft_band, kaiser(cnt.fs,2));

figure('Name',['FFT Epochs'])
grid_plot(epo_spec, mnt, opt_spec)

epo_spec_avg=proc_average(epo_spec);
max_y = find(epo_spec.t==plot_classes(i));
ylevel_1 = epo_spec_avg.x(max_y,ind_1);
ylevel_2 = epo_spec_avg.x(max_y,ind_2);
figure('Name',['FFT Epochs channel ', 'Mod.freq = ',int2str(plot_classes(i)),' Hz'])
    subplot(2,2,1)
    plotChannel(epo_spec,plot_chan{1})
    text(plot_classes(i),ylevel_1,[' \bf \leftarrow ',int2str(plot_classes(i)),' Hz'],'Rotation',40,'FontSize',9,'VerticalAlignment','middle','Color',[1 0 0])
     subplot(2,2,3)
    plotChannel(epo_spec,plot_chan{2})
    text(plot_classes(i),ylevel_2,[' \bf \leftarrow ',int2str(plot_classes(i)),' Hz'],'Rotation',40,'FontSize',9,'VerticalAlignment','middle','Color',[1 0 0])
    
    subplot(2,2,2)
    plotChannel(epo_spec_sub,plot_chan{1})
    text(plot_classes(i),ylevel_1,[' \bf \leftarrow ',int2str(plot_classes(i)),' Hz'],'Rotation',40,'FontSize',9,'VerticalAlignment','middle','Color',[1 0 0])
 
    subplot(2,2,4)
    plotChannel(epo_spec_sub,plot_chan{2})
    text(plot_classes(i),ylevel_2,[' \bf \leftarrow ',int2str(plot_classes(i)),' Hz'],'Rotation',40,'FontSize',9,'VerticalAlignment','middle','Color',[1 0 0])
    
    
    epo_spectr= proc_spectrum(epo_sub, [plot_classes(i)-1 plot_classes(i)+1], kaiser(cnt.fs,2));
    epo_avg_spectr = proc_average(epo_spectr);
    epo_spec_max1(i) = max(epo_avg_spectr.x(:,ind_1));
    epo_spec_max2(i) = max(epo_avg_spectr.x(:,ind_2));
    
    epo_band = proc_Bandpower(epo,[plot_classes(i)-1 plot_classes(i)+1]);

    epo_avg_band = proc_average(epo_band);

    epo_band_chan_1(i) = epo_avg_band.x(:,ind_1);
    epo_band_chan_2(i) = epo_avg_band.x(:,ind_2);
    
    
    
end

figure('Name','Tuningfunctions')
subplot(2,1,1)
plot(plot_classes,epo_band_chan_1,'--rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',8)
grid on
xlabel('Modulation Frequency [Hz]')
ylabel('Band Power')
title(plot_chan{1})
subplot(2,1,2)
plot(plot_classes,epo_band_chan_2,'--rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',8)
grid on
xlabel('Modulation Frequency [Hz]')
ylabel('Band Power')
title(plot_chan{2})

figure('Name','Tuningfunctions 2')
subplot(2,1,1)
plot(plot_classes,epo_spec_max1,'--rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',8)
grid on
xlabel('Modulation Frequency [Hz]')
ylabel('max amp [dB]')
title(plot_chan{1})
subplot(2,1,2)
plot(plot_classes,epo_spec_max2,'--rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',8)
grid on
xlabel('Modulation Frequency [Hz]')
ylabel('max amp [dB]')
title(plot_chan{2})