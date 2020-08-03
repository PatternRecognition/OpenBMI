function fil = find_file(file,ext)

poi = 0;

fil = [file int2str(poi) '.' ext];

while exist(fil,'file')
  poi = poi+1;
  fil = [file int2str(poi) '.' ext];
end
