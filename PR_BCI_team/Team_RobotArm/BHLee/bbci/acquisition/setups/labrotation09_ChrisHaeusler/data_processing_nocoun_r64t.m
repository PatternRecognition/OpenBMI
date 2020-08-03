EEG_RAW_DIR= 'C:\uni\lab rotation\bbciRaw\';
EEG_MAT_DIR= 'C:\uni\lab rotation\bbciMat\';
targetWindows = {200 500 1000 1500 2000};
clear results subject
subject(1).name='VPnh_vis_nocount';
subject(1).path='Thomas_09_10_23/Thomas_vis_nocount'; 
subject(2).name='VPnh_tact_nocount';
subject(2).path='Thomas_09_10_23/Thomas_tact_nocount';
subject(3).name='VPmk_vis_nocount';
subject(3).path='sophie_09_10_30/sophie_vis_nocount';
subject(4).name='VPmk_tact_nocount';
subject(4).path='sophie_09_10_30/sophie_tact_nocount';
subject(5).name='VPgao_vis_nocount';
subject(5).path='chris_09_11_17/chris_vis_nocount';
subject(6).name='VPgao_tact_nocount';
subject(6).path='chris_09_11_17/chris_tact_nocount';
subject(7).name='VPiac_vis_nocount';
subject(7).path='nico_09_11_12/nico_vis_nocount';
subject(8).name='VPiac_tact_nocount';
subject(8).path='nico_09_11_12/nico_tact_nocount';
subject(9).name='rithwick_vis_nocount';
subject(9).path='rithwick_09_11_05/rithwick_vis_nocount';   
subject(10).name='rithwick_tact_nocount';
subject(10).path='rithwick_09_11_05/rithwick_tact_nocount';

indSubject = 1
for sub=1:length(subject),
    numTargets = 0;
    numHits = zeros(1,length(targetWindows));
    subject(sub).path
    [mrko, FS] = eegfile_readBVmarkers(subject(sub).path);
   
    mrkCopy = mrko;
    jj = strmatch('R 64',mrkCopy.desc);
    mrkCopy.pos(jj)= 0;
        
        
    ii= find(diff(mrkCopy.pos)<10 & diff(mrkCopy.pos)>0 );
    mrko.pos(ii)= [];
    mrko.desc(ii)= [];

     iCueStart= strmatch('S  1',mrko.desc);
     nTrials= length(iCueStart);
     iCueStart(nTrials + 1) = length(mrko.desc)+1;
     nTargets= 4;
     nRepetitions= 5;
     nStimuli= nTrials*nTargets*nRepetitions;
 
     fcn= inline('str2num(x(end))','x');
     ptr= 0;
     
     for ii= 1:nTrials,         
         clear trial
         trial.pos = mrko.pos(iCueStart(ii):iCueStart(ii+1)-1);
         trial.desc = mrko.desc(iCueStart(ii):iCueStart(ii+1)-1);
         
         target= trial.desc(2);
         iTrialStart= strmatch('S100',trial.desc);
         if length(iTrialStart) > 1,
             error('only 1 S100 expected');
         end
         
         
         
         targets= strmatch(target,trial.desc(iTrialStart:end));
         numTargets = numTargets + length(targets);
         for jj=1:length(targetWindows),
             for kk=1:length(targets),
                searchWindow = trial.desc(find(trial.pos > trial.pos(targets(kk)) & trial.pos < trial.pos(targets(kk))+ targetWindows{jj}));
                hits = strmatch('R 64', searchWindow);
                if length(hits) > 0,
                    numHits(jj) = numHits(jj) + 1;
                end
             end;
         end;

    end;
    %numTargets
    %numHits
    error = 1 - (numHits / numTargets)
    if mod(sub,2)
        indSubject
        results(indSubject,1,:) = error;
    else
        indSubject
        results(indSubject,2,:) = error;
        indSubject = indSubject + 1;
    end
 
end;
close all
figure
bar(squeeze(results(:,1,:)))
title('Visual Response Error')
ylabel('Response Error %')
xlabel('Subject')   

figure
bar(squeeze(results(:,2,:)))
ylabel('Response Error %')
xlabel('Subject')   
set(gca,'xticklabel',xLab)

title('Tactile Response Error')

figure
hold all
bar(mean(results(:,:,5)), 0.2)
errorbar(mean(results(:,:,5)), std(results(:,:,5)),'r','linestyle', 'none', 'linewidth', 2)
hold off
xLab = {'Visual Stimulus';'Tactile Stimulus';}
set(gca,'xticklabel',xLab)
set(gca,'XTick',[1 2]) 
ylabel('Average Response Error % with error bars')
xlabel('Stimulus Method')   
title('Average Response Error')

