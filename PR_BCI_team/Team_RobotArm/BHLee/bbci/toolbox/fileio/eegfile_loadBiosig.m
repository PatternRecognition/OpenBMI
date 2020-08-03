function [cnt,mrk,mnt,HDR]= eegfile_loadBiosig(sub_dir,filt)
% EEGFILE_LOADBIOSIG - loads raw bbci data into matlab structures using the BioSig toolbox
% 
% Synopsis:
%   [CNT, MRK, MNT]= eegfile_loadBioSig(SUB_DIR, FILT)
%
% Arguments:
%   SUB_DIR: subdirectory of '/home/neuro/data/BCI/bbciRaw' containing the
%            data
%   FILT: experimental paradigm, either 'real' or 'imag_lett'
%
% Returns:
%   DAT: structure of continuous or epoched signals
%   MRK: marker structure
%   MNT: electrode montage structure
%
% See also: eegfile_*
%
% Legal Note:
%   "BioSig for Octave and Matlab" is available from http://biosig.sf.net
%
%   This library is free software; you can redistribute it and/or
%   modify it under the terms of the GNU Library General Public
%   License as published by the Free Software Foundation; either
%   Version 2 of the License, or (at your option) any later version.
%
%   This library is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%   Library General Public License for more details.
%
%   You should have received a copy of the GNU Library General Public
%   License along with this library; if not, write to the
%   Free Software Foundation, Inc., 59 Temple Place - Suite 330,
%   Boston, MA  02111-1307, USA.
%
% Author(s): Copyright (C) 2007 by Alois Schloegl
%

if 0, 
	% loads RAW SEASON2 data (128#)
	pfad2 = '/home/dropbox/IDA/blanker/';
  	[a1,a2] = season2player(sub_dir);
	tmp = sprintf('season2_%02i_%02i_%02i',a2);
	x1  = season2player(tmp,1);
	x2  = season2player(tmp,2);
	SW  = strncmp(sub_dir,x2,length(x2));
	p   = [pfad2,tmp];
	fn  = dir([p,filesep,filt,'*.vhdr']);
%	for k=length(fn):-1:1,
	for k=1:length(fn),
		F(k).name = fullfile(p,fn(k).name); 
	end; 	
	CHAN = (1:59)+SW*64; 
else
	pfad2 = '/home/neuro/data/BCI/bbciRaw'; 
	p   = [pfad2,filesep,sub_dir];
	f0  = dir([p,filesep,filt,'*.vhdr']);
        ix2 = [];
        for k3=1:length(f0)
        	if isempty(strfind(f0(k3).name,'eog_mono'))
        		ix2=[ix2,k3];
        	end;
        end;
        fn = f0(ix2);
	for k=1:length(fn),
		F(k).name = fullfile(p,fn(k).name); 
	end; 	
	CHAN = (1:59); 
end; 

cnt=[]; mrk=[]; mnt=[]; HDR=[];
if ~exist('sopen','file')
	fprintf(2,'Error eegfile_loadBiosig: sload not found.\n You need to install BioSig for Octave and Matlab http://biosig.sf.net\n'); 
	return; 
end; 

[s,HDR]=sload(F,CHAN,'OVERFLOWDETECTION','OFF'); 

D = 10;
cnt.clab  = HDR.Label(CHAN)'; 
cnt.fs    = HDR.SampleRate/D; 
cnt.file  = fn; 
[p,f1,e]  = fileparts(F(1).name);
[p,f2,e]  = fileparts(p);
cnt.title = [f1,filesep,f2]; 
cnt.x     = rs(s,D,1);
%cnt.x    = s(1:D:end,:);

ix = (HDR.EVENT.TYP<4); 
mrk.pos = round(HDR.EVENT.POS(ix)'/D); 
%mrk.pos = floor(HDR.EVENT.POS(ix)'/D); 
%mrk.pos = ceil(HDR.EVENT.POS(ix)'/D); 
mrk.toe = HDR.EVENT.TYP(ix)'; 
%mrk.pos = HDR.TRIG'; 
%mrk.typ = HDR.Classlabel'; 
mrk.fs  = HDR.SampleRate/D; 
mrk.y   = full(sparse(mrk.toe,1:length(mrk.toe),1)); 
mrk.className =  {'left'  'right'  'foot'};

mnt.pos_3d = HDR.ELEC.XYZ';
mnt.clab = HDR.SampleRate; 
