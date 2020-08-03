function bbci_bet_create_figure(picture);
% ALLOWS FIGURE CREATION AS MATLAB FILE
[im,cm] = imread(picture);

c = strfind(picture,'.');
picture = picture(1:c(end)-1);

fid = fopen([picture '.m'],'w');

fprintf(fid,'function [im,cm] = %s\n\n',picture);
fprintf(fid,'im = uint8(zeros(%d,%d,3));\n\n',size(im,1),size(im,2));

for i = 1:3
  fprintf(fid,'im(:,:,%d) = [',i);
  for j = 1:size(im,1)
    fprintf(fid,'%d,',double(im(j,1:end-1,i)));
    fprintf(fid,'%d',double(im(j,end,i)));
    if j==size(im,1)
      fprintf(fid,'];\n\n');
    else
      fprintf(fid,';...\n');
    end
  end
end

fprintf(fid,'cm=[];\n\n');

fclose(fid);

