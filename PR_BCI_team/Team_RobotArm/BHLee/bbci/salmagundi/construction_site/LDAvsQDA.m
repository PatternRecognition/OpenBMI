function LDAvsQDA(n1,n2,EW1,EW2,COV1,COV2,varargin)
%LDAvsQDA(n1,n2,EW1,EW2,COV1,COV2, <outliers..>)

val1 = chol(COV1)'*randn(2,n1)+repmat(EW1,[1 n1]);
val2 = chol(COV2)'*randn(2,n2)+repmat(EW2,[1 n2]);

sqbest = pinv(COV2) - pinv(COV1);
wbest = 2*pinv(COV1)*EW1 - 2*pinv(COV2)*EW2;
bbest = -EW1'*pinv(COV1)*EW1+EW2'*pinv(COV2)*EW2 + log(det(COV2)/det(COV1));
[U1,V1] = eig(COV1);
[U2,V2] = eig(COV2);
U1 = U1*V1;
U2 = U2*V2;


EW1ges = mean(val1,2);
EW2ges = mean(val2,2);
COV1ges = cov(val1');
COV2ges = cov(val2');
COVges = ((n1-1)*COV1ges + (n2-1)*COV2ges)/(n1+n2-1);
[U,V] =eig(COVges);
ULges = U*V;
[U,V] = eig(COV1ges);
UQ1ges = U*V;
[U,V] = eig(COV2ges);
UQ2ges = U*V;

wLges = 2*pinv(COVges)*(EW1ges-EW2ges);
bLges = -0.5*(EW1ges+EW2ges)'*wLges;

sqQges = pinv(COV2ges) - pinv(COV1ges);
wQges = 2*pinv(COV1ges)*EW1ges - 2*pinv(COV2ges)*EW2ges;
bQges =  -EW1ges'*pinv(COV1ges)*EW1ges+EW2ges'*pinv(COV2ges)*EW2ges +log(det(COV2ges)/det(COV1ges));

if length(varargin)>0
  redO = [];
  blueO = [];
  for i=1:length(varargin)
    c = varargin{i};
    if c(1) ==  1
        redO = cat(2,redO,c(2:3)');
    else
        blueO = cat(2,blueO,c(2:3)');
    end
  end
  val1 = cat(2,val1,redO);
  val2 = cat(2,val2,blueO);
end

plot(val1(1,1:n1),val1(2,1:n1),'r+')
hold on;plot(val2(1,1:n2),val2(2,1:n2),'b+')
plot(val1(1,n1+1:end),val1(2,n1+1:end),'ro');
plot(val2(1,n2+1:end),val2(2,n2+1:end),'bo');

xmin=getfield(get(gca,'XLim'),{1});
ymin=getfield(get(gca,'YLim'),{1});
xmax=getfield(get(gca,'XLim'),{2});
ymax=getfield(get(gca,'YLim'),{2});
for x = 0:100
    xx = xmin+x*(xmax-xmin)/100;
    for y = 0:100
        yy= ymin+y*(ymax-ymin)/100;
        v(y+1,x+1) = [xx,yy]*sqbest*[xx;yy] + wbest'*[xx;yy] + bbest;
    end
end

contour(xmin:0.01*(xmax-xmin):xmax,ymin:0.01*(ymax-ymin):ymax,v,[0 0],'k');

h_true = line(EW1(1)+[-1;1]*U1(1,:), EW1(2)+[-1;1]*U1(2,:));
h2     = line(EW2(1)+[-1;1]*U2(1,:), EW2(2)+[-1;1]*U2(2,:));
set([h_true h2], 'Color','k')
for x = 0:100
    xx = xmin+x*(xmax-xmin)/100;
    for y = 0:100
        yy= ymin+y*(ymax-ymin)/100;
        v(y+1,x+1) = wLges'*[xx;yy] + bLges;
    end
end
contour(xmin:0.01*(xmax-xmin):xmax,ymin:0.01*(ymax-ymin):ymax,v,[0 0],'c');
h_LDA = line(EW1ges(1)+[-1;1]*ULges(1,:), EW1ges(2)+[-1;1]*ULges(2,:));
h2    = line(EW2ges(1)+[-1;1]*ULges(1,:), EW2ges(2)+[-1;1]*ULges(2,:));
set([h_LDA h2], 'Color','c')

for x = 0:100
    xx = xmin+x*(xmax-xmin)/100;
    for y = 0:100
        yy= ymin+y*(ymax-ymin)/100;
        v(y+1,x+1) = [xx,yy]*sqQges*[xx;yy] + wQges'*[xx;yy] + bQges;
    end
end
contour(xmin:0.01*(xmax-xmin):xmax,ymin:0.01*(ymax-ymin):ymax,v,[0 0],'y');
h_QDA = line(EW1ges(1)+[-1;1]*UQ1ges(1,:), EW1ges(2)+[-1;1]*UQ1ges(2,:));
h2    = line(EW2ges(1)+[-1;1]*UQ2ges(1,:), EW2ges(2)+[-1;1]*UQ2ges(2,:));
set([h_QDA h2], 'Color','y')

if length(varargin)==0,
  legend([h_true(1) h_LDA(1) h_QDA(1)], 'true','LDA','QDA', -1);
  hold off;
  return;
end

EW1ges = mean(val1,2);
EW2ges = mean(val2,2);
COV1ges = cov(val1');
COV2ges = cov(val2');
COVges = ((size(val1,2)-1)*COV1ges + (size(val2,2)-1)*COV2ges)/(n1+n2-1);
[U,V] =eig(COVges);
ULges = U*V;
[U,V] = eig(COV1ges);
UQ1ges = U*V;
[U,V] = eig(COV2ges);
UQ2ges = U*V;

wLges = 2*pinv(COVges)*(EW1ges-EW2ges);
bLges = -0.5*(EW1ges+EW2ges)'*wLges;

sqQges = pinv(COV2ges) - pinv(COV1ges);
wQges = 2*pinv(COV1ges)*EW1ges - 2*pinv(COV2ges)*EW2ges;
bQges =  -EW1ges'*pinv(COV1ges)*EW1ges+EW2ges'*pinv(COV2ges)*EW2ges +log(det(COV2ges)/det(COV1ges));

for x = 0:100
    xx = xmin+x*(xmax-xmin)/100;
    for y = 0:100
        yy= ymin+y*(ymax-ymin)/100;
        v(y+1,x+1) = wLges'*[xx;yy]+bLges;
    end
end
contour(xmin:0.01*(xmax-xmin):xmax,ymin:0.01*(ymax-ymin):ymax,v,[0 0],'m');
h_LDA_o = line(EW1ges(1)+[-1;1]*ULges(1,:), EW1ges(2)+[-1;1]*ULges(2,:));
h2      = line(EW2ges(1)+[-1;1]*ULges(1,:), EW2ges(2)+[-1;1]*ULges(2,:));
set([h_LDA_o h2], 'Color','m')

for x = 0:100
    xx = xmin+x*(xmax-xmin)/100;
    for y = 0:100
        yy= ymin+y*(ymax-ymin)/100;
        v(y+1,x+1) = [xx,yy]*sqQges*[xx;yy] + wQges'*[xx;yy]+bQges;
    end
end
contour(xmin:0.01*(xmax-xmin):xmax,ymin:0.01*(ymax-ymin):ymax,v,[0 0],'g');
h_QDA_o = line(EW1ges(1)+[-1;1]*UQ1ges(1,:), EW1ges(2)+[-1;1]*UQ1ges(2,:));
h2      = line(EW2ges(1)+[-1;1]*UQ2ges(1,:), EW2ges(2)+[-1;1]*UQ2ges(2,:));
set([h_QDA_o h2], 'Color','g')

legend([h_true(1) h_LDA(1) h_LDA_o(1) h_QDA(1) h_QDA_o(1)], ...
       'true','LDA','LDA w/Out','QDA','QDA w/Out', -1);
hold off
