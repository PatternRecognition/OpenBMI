function mrk = getMarkerFromHeader(hdr,fs) 
  mrk.fs  = fs ; 
  mrk.pos = [] ;
  mrk.toe = [] ;
  mrk.className = {} ;
  mrk.y   = [] ;
  
  tag= '[Events]';
  is= strfind(hdr.Header, tag) + length(tag);
  
  markerTags = {'Augen','HV Start','PHV', 'Ansprache','Artefakt','Photic'};
  
  eolPos = strfind(hdr.Header,char(10)) ;
  for markerTagId = 1:length(markerTags) ,
    markerTag = markerTags{markerTagId} ;
    tagPos = strfind(hdr.Header, [char(9) markerTag])+1 ;
    
    numTagsFound = length(tagPos) ;
    if numTagsFound > 0,
      switch markerTag,
       case {'Photic', 'HV Start', 'Augen', 'Ansprache', 'Artefakt'}
	for foundId = 1: numTagsFound ,
	  eolBehindTagId = min(find(eolPos>tagPos(foundId))) ;
	  marker   = hdr.Header(tagPos(foundId):eolPos(eolBehindTagId)-1);
	  classIdx = find(strcmp(marker, mrk.className)==1) ; 
	  % if marker not yet in classNames-list the append a new
	  % class to the marker structure   
	  if isempty(classIdx) ,
	    mrk.className{end+1} = marker ;
	    mrk.y = [mrk.y ; zeros(1,size(mrk.y,2))] ;
	    classIdx = size(mrk.y,1) ;
	  end ;
	  mrk.y = [mrk.y  zeros(size(mrk.y,1),1)] ;
	  wholeLine = hdr.Header((eolPos(eolBehindTagId-1)+1):eolPos(eolBehindTagId)-1);
	  tabPosInLine = strfind(wholeLine,char(9)) ;
	  mrk.pos(end+1) = str2num(wholeLine((tabPosInLine(1)+1):(tabPosInLine(2)-1)));
	  mrk.y(classIdx, end) = 1.0;
	  mrk.toe(end+1) = str2num(wholeLine((tabPosInLine(2)+1):(tabPosInLine(3)-1)));
	end ;
       case 'PHV',
	for foundId = 1: numTagsFound ,
	  eolBehindTagId = min(find(eolPos>tagPos(foundId))) ;
	  marker   = hdr.Header(tagPos(foundId):eolPos(eolBehindTagId)-1);
	  % marker "PHV Start" implies "HV End" at position-1 
	  if strcmp(marker, 'PHV Start'),
	    tmpMarker = 'HV End' ;
	    tmpClassIdx = find(strcmp(tmpMarker, mrk.className)==1) ; 
	    % if "HV End" not yet in classNames-list the append a new
	    % class to the marker structure   
	    if isempty(tmpClassIdx) ,
	      mrk.className{end+1} = tmpMarker ;
	      mrk.y = [mrk.y ; zeros(1,size(mrk.y,2))] ;
	      tmpClassIdx = size(mrk.y,1) ;
	    end ;
	    mrk.y = [mrk.y  zeros(size(mrk.y,1),1)] ;
	    wholeLine = hdr.Header((eolPos(eolBehindTagId-1)+1):eolPos(eolBehindTagId)-1);
	    tabPosInLine = strfind(wholeLine, char(9)) ;
	    mrk.pos(end+1) = str2num(wholeLine((tabPosInLine(1)+1):(tabPosInLine(2)-1)))-1;
	    mrk.y(tmpClassIdx, end) = 1.0;
	    mrk.toe(end+1) = str2num(wholeLine((tabPosInLine(2)+1):(tabPosInLine(3)-1)));
	 
	    classIdx = find(strcmp(marker, mrk.className)==1) ; 
	    % if marker not yet in classNames-list the append a new
	    % class to the marker structure   
	    if isempty(classIdx) ,
	      mrk.className{end+1} = marker ;
	      mrk.y = [mrk.y ; zeros(1,size(mrk.y,2))] ;
	      classIdx = size(mrk.y,1) ;
	    end ;
	    mrk.y = [mrk.y  zeros(size(mrk.y,1),1)] ;
	    wholeLine = hdr.Header((eolPos(eolBehindTagId-1)+1):eolPos(eolBehindTagId)-1);
	    tabPosInLine = strfind(wholeLine, char(9)) ;
	    mrk.pos(end+1) = str2num(wholeLine((tabPosInLine(1)+1):(tabPosInLine(2)-1)));
	    mrk.y(classIdx, end) = 1.0;
	    mrk.toe(end+1) = str2num(wholeLine((tabPosInLine(2)+1):(tabPosInLine(3)-1)));
	  end ;
	end ;
	% last marker "PHV *" implies "PHV End" marker
	marker = 'PHV End' ;
	classIdx = find(strcmp(marker, mrk.className)==1) ; 
	% if marker not yet in classNames-list the append a new
	% class to the marker structure   
	if isempty(classIdx) ,
	  mrk.className{end+1} = marker ;
	  mrk.y = [mrk.y ; zeros(1,size(mrk.y,2))] ;
	  classIdx = size(mrk.y,1) ;
	end ;
	mrk.y = [mrk.y  zeros(size(mrk.y,1),1)] ;
	wholeLine = hdr.Header((eolPos(eolBehindTagId-1)+1):eolPos(eolBehindTagId)-1);
	tabPosInLine = strfind(wholeLine, char(9)) ;
	mrk.pos(end+1) = str2num(wholeLine((tabPosInLine(1)+1):(tabPosInLine(2)-1)));
	mrk.y(classIdx, end) = 1.0;
	mrk.toe(end+1) = str2num(wholeLine((tabPosInLine(2)+1):(tabPosInLine(3)-1)));

      end ; % end case
    end ; % end if 
    
  end ; % end loop for makerTags

