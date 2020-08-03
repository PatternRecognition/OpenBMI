% Convert and investigate data 



%% CONVERT
sbj = VP_CODE;
% grd= sprintf(['scale,_,F5,F3,Fz,F4,F6,_,legend\n' ...
%               'FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8\n' ...
%               'T7,C5,C3,C1,Cz,C2,C4,C6,T8\n' ...
%               'P7,P5,P3,P1,Pz,P2,P4,P6,P8\n' ...
%               'PO9,PO7,PO3,O1,Oz,O2,PO4,PO8,PO10']);

new_fs = 500;
file= [TODAY_DIR '/' filelist sbj];

try
  hdr= eegfile_readBVheader([file '*']);
catch
  fprintf('\n*** file %s not found.\n\n', file);
  continue
end

%% load and process EEG channels
% *** TODO : filter ***
Wps= [42 49]/hdr.fs*2;
[n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 40);
[filt.b, filt.a]= cheby2(n, 50, Ws);

[cnt, mrk_orig]= eegfile_readBV([file '*'], 'fs',new_fs, ...
                                'filt',filt);

%% Re-referencing to linked mastoids
% A= eye(length(cnt.clab));
% iA2= chanind(cnt.clab,'A2');
% A(iA2,:)= -0.5;
% A(:,iA2)= [];
% cnt= proc_linearDerivation(cnt, A);

%% Markers
mrk = mrk_defineClasses(mrk_orig, {26,42,58,74,90,106,122,138,154,170,186,202,8,248; ...
                            '26Hz','42Hz','58Hz', '74Hz', '90Hz', '106Hz', '122Hz', '138Hz', '154Hz', '170Hz', '186Hz', '202Hz', 'CWL','TrialStart'});
% TODO

%% Montage
mnt= getElectrodePositions(cnt.clab);
mnt= mnt_setGrid(mnt, grd);
mnt= mnt_excenterNonEEGchans(mnt, 'E*');

%% Save
fs_orig= mrk_orig.fs;
var_list= {'fs_orig',fs_orig, 'mrk_orig',mrk_orig, 'hdr',hdr, 'bbci',bbci};
eegfile_saveMatlab(file, cnt, mrk, mnt, ...
                   'channelwise',1, ...
                   'format','int16', ...
                   'resolution', NaN, ...
                   'vars', var_list);

%% INVESTIGATION
