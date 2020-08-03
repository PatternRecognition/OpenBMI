EEG_RAW_DIR= 'C:\uni\lab rotation\bbciRaw\';
EEG_MAT_DIR= 'C:\uni\lab rotation\bbciMat\';
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

for sub=1:length(subject),
    hdr= eegfile_readBVheader(subject(sub).path);
    Wps= [40 49]/hdr.fs*2;
    [n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
    [filt.b, filt.a]= cheby2(n, 50, Ws);
    bip_list= {{'Fp2','EOGvu','EOGv'},{'F9','F10','EOGh'}};
    ld= procutil_biplist2projection(hdr.clab, bip_list);
    %[cnt, mrko]= eegfile_readBV([file '*'], ...
    [cnt, mrko]= eegfile_loadBV([file '*'], ...
       'fs',100, ...
       'filt',filt, ...
       'linear_derivation',ld);

    jj = strmatch('R 64',mrko.desc);
    mrko.desc(jj)= [];
    mrko.pos(jj)= [];
    ii= find(diff(mrko.pos)<10);
    mrko.desc(ii)= [];
    mrko.pos(ii)= [];
    iCueStart= strmatch('S  1',mrko.desc);
    nTrials= length(iCueStart);
    nTargets= 4;
    nRepetitions= 5;
    nStimuli= nTrials*nTargets*nRepetitions;

    clear mrk
    mrk.pos= zeros(1, nStimuli);
    mrk.toe= zeros(1, nStimuli);
    mrk.target= zeros(1, nStimuli);
    mrk.fs= mrko.fs;
    mrk.className= {'target','nontarget'};

    fcn= inline('str2num(x(end))','x');
    ptr= 0;
    for ii= 1:nTrials,
        ii
        base_idx= iCueStart(ii);
        target= mrko.desc{base_idx+1};
        if ~strcmp(mrko.desc{base_idx+6},'S100'),
            error('S100 expected');
        end
        idx_from= base_idx + 6 + [1:nTargets*nRepetitions];
        if ~strcmp(mrko.desc{idx_from(end)+1},'S  4'),
            error('S  4 expected');
        end
        idx_to= ptr + [1:nTargets*nRepetitions];
        mrk.pos(idx_to)= mrko.pos(idx_from);
        mrk.toe(idx_to)= apply_cellwise2(mrko.desc(idx_from), fcn);
        mrk.target(idx_to)= fcn(target);
        ptr= ptr + nTargets*nRepetitions;
    end
    mrk.y= [mrk.toe==mrk.target; mrk.toe~=mrk.target];

    mnt= getElectrodePositions(cnt.clab);
    grd= sprintf(['Fp2,scale,F3,Fz,F4,legend,F10\n' ...
        'C5,C3,C1,Cz,C2,C4,C6\n' ...
        'P5,P3,P1,Pz,P2,P4,P6\n' ...
        'P9,PO7,O1,Oz,O2,PO8,P10']);
    mnt= mnt_setGrid(mnt, grd);

    vars= {'hdr',hdr, 'mrk_orig',mrko},
    eegfile_saveMatlab(subject(sub).path, cnt, mrk, mnt, 'vars',vars);

    %eegfile_concatMatlab
end;