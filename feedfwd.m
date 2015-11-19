function [Y,o1_n] = feedfwd(W,inex,o1)

global nwts nhid nin nout;

W1 = reshape( W(1:nhid*(nin+1+nhid)),nhid,nin+1+nhid );
    
W2 = reshape( W(nhid*(nin+1+nhid)+1:end),nout,nhid+1 );
    
u1 = W1*[inex;1;o1];

o1_n = (2./(1+exp(-2*u1)) - 1);
    
Y = W2*[o1_n;1];
    


