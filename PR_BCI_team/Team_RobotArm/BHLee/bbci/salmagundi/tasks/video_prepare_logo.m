subdir= [TEX_DIR 'pics/logos/'];
%logofile= [subdir 'bbci_logo_trans.png'];
logofile= [subdir 'bbci_typo_trans.png'];
[A,cmap,alpha]= imread(logofile);

imagesc(A); axis equal;
for yy= 1:size(A,1),
  for xx= 1:size(A,2),
    if alpha(yy,xx)==0,
      A(yy,xx,:)= 255;
    end
  end
end

for yy= 1:size(A,1),
  for xx= 1:size(A,2),
    if all(A(yy,xx,:)==A(yy,xx,1)) & A(yy,xx,1)<255,
      A(yy,xx,:)= uint8( 255 - double(A(yy,xx,1)) );
    end
  end
end

for yy= 1:size(A,1),
  for xx= 1:size(A,2),
    if alpha(yy,xx)==0,
      A(yy,xx,:)= 0;
    end
  end
end

AA= A;
B= rgb2hsv(A);
for yy= 1:size(A,1),
  for xx= 1:size(A,2),
    if abs(B(yy,xx,1))>0.9 & B(yy,xx,2)>0.2 & B(yy,xx,2)<0.9,
      pix= squeeze(B(yy,xx,[1 3 2]));
      pix(2)= 1;
      AA(yy,xx,:)= hsv2rgb(pix');
    end
  end
end
imagesc(AA); axis equal;

A= AA(1:361, 400:end, :);
alpha= alpha(1:361, 400:end, :);
imagesc(A); axis equal;

imwrite(A, [subdir 'bbci_black_trans.png'], 'png', 'alpha',alpha);
