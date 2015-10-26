function [xkk,Skk] = SCKFwt(xkk,Skk,signal,o1,yk)

global Rw lambda nwts;

nx = nwts;

nPts = 2*nx;

CPtArray = sqrt(nx)*[eye(nx)  -eye(nx)];      %Cubature Point Array

Xi =  repmat(xkk,1,nPts) + Skk*CPtArray/sqrt(lambda);
    
Zi = MstEq(Xi,signal,o1);
    
zkk1 = sum(Zi,2)/nPts; 

nz = length(zkk1); %dim. of mst vector

Z = (Zi-repmat(zkk1,1,nPts))/sqrt(nPts);  

X = (Xi-repmat(xkk,1,nPts))/sqrt(nPts);

[foo,S] = qr([Z sqrt(Rw); X zeros(nx,nz)]',0);

S = S';

A = S(1:nz,1:nz);   % Square-root Innovations Covariance

B = S(nz+1:end,1:nz);

C = S(nz+1:end,nz+1:end);

G = B/A;          % Cubature Kalman Gain  

xkk = xkk + G*(yk-zkk1);  
    
Skk = C;


%==========================================================================

function YArray = MstEq(WArray,inex,o1)

global nhid nin nout nwts;

for i = 1:2*nwts
    
    W1 = reshape( WArray(1:nhid*(nin+1+nhid),i),nhid,nin+1+nhid );
    
    W2 = reshape( WArray(nhid*(nin+1+nhid)+1:nwts,i),nout,nhid+1 );
    
    u1 = W1*[inex;1;o1];

    o1_n = (2./(1+exp(-2*u1)) - 1);    
    
    YArray(:,i) = W2*[o1_n;1];   
    
end;

