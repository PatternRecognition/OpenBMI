function [code, subject_list]= getSubjectCode(sbj, varargin)
%GETSUBJECTCODE - Get code of a BBCI subject
%
%Synopsis:
% CODE= getSubjectCode(SBJ)
%
%Arguments:
% SBJ: String. Name of a subject or subdirectory of EEG_RAW_DIR.
%      For names like 'VPcm' code 'cm' is returned.
%
%Returns
% CODE: two-letter subject code

% Author(s): Benjamin Blankertz, Oct 2006


global EEG_RAW_DIR EEG_CFG_DIR

iu= find(sbj=='_');
if ~isempty(iu),
  %% first arg is subdir
  sbj= sbj(1:iu-1);
end

if length(sbj)==4 & strncmp(sbj, 'VP', 2),
  code= sbj(3:4);
  return;
end
  
dd= dir([EEG_RAW_DIR '/*_*_*_*']);
if length(dd)<200,
  %% For my laptop. Use the following line to update.
  %% ssh $VNC "cd /home/data/BCI/bbciRaw; ls -d *_*_*_*" > $BCI/data/config/bbciRaw_dir 
  dn= textread([EEG_CFG_DIR 'bbciRaw_dir'], '%s');
else
  dn= {dd.name};
end

iValid1= find(apply_cellwise2(dn, inline('ismember(x(2),97:122)','x')));
iValid2= find(~apply_cellwise2(dn, inline('strncmp(x,''Vp'',2)', 'x')));
dn= dn(intersect(iValid1,iValid2));

nSub= 0;
subject_list= {};
date_list= {};
for nn= 1:length(dn),
  is= min(find(dn{nn}=='_'));
  sn= dn{nn}(1:is-1);
  if ~ismember(sn, subject_list),
    nSub= nSub + 1;
    subject_list{nSub}= sn;
    date_list{nSub}= dn{nn}(is+[1 2 4 5 7 8]);
  end
end
[date_list, si]= sort(date_list);
subject_list= subject_list(si);

idx= strmatch(sbj, subject_list, 'exact');
available_leads= 'azyxwvu';
i1= ceil(idx/26);
i2= mod(idx-1, 26);
code= [available_leads(i1) char('a'+i2)];
