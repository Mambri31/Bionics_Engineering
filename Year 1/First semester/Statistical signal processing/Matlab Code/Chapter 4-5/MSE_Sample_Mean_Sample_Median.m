%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola la stima del parametro A (valor medio e
% mediana dei dati) mediante gli stimatori Sample Mean e Sample Median
% e ne calcola l'MSE per rumore Gaussiano oppure di Laplace
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nvect=[2:2:1000];   % Sample size
Nd=length(Nvect);
M=10^4;             % Numero di prove Monte Carlo dell'esperimento

A=2;                % valor medio v.a. Gaussiana/Laplace
varx=1;             % varianza v.a. Gaussiana (G)
eta=sqrt(varx/2);   % parametro della v.a. di Laplace (L) in modo che G ed L abbiano la stessa potenza

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 h=waitbar(0,'Please wait...');
 
for i=1:Nd
    
    waitbar(i/Nd,h); 
    N=Nvect(i);
    X=sqrt(varx)*randn(N,M)+A; % Si generano N campioni di una v.a. Gaussiana
    U=rand(N,M);
    XE=-eta*log(U); 
    S=sign(rand(N,M)-0.5);
    XL=S.*XE+A;
    vmX=mean(X);
    medX=median(X);
    vmL=mean(XL);
    medL=median(XL);
    MSEvmX(i)=sum((vmX-A).^2)/M;
    MSEmedX(i)=sum((medX-A).^2)/M;
    MSEvmL(i)=sum((vmL-A).^2)/M;
    MSEmedL(i)=sum((medL-A).^2)/M;

end

close(h)

CRB_G=varx./Nvect;
CRB_L=(eta^2)./Nvect;

figure (1);
loglog(Nvect,CRB_G,'green', 'LineWidth',2);grid on;
hold on;
loglog(Nvect,MSEvmX,'blue', 'LineWidth',2);
loglog(Nvect,MSEmedX,'red', 'LineWidth',2);
legend('CRB(A)', 'MSE Sample Mean','MSE Sample Median');
title('Estimate of A - Gaussian noise');xlabel('Sample size (N)');ylabel('Mean Square Error (MSE)')

figure (2);
loglog(Nvect,CRB_L,'green', 'LineWidth',2);grid on;
hold on;
loglog(Nvect,MSEvmL,'blue', 'LineWidth',2);
loglog(Nvect,MSEmedL,'red', 'LineWidth',2);
legend('CRB(A)', 'MSE Sample Mean','MSE Sample Median');
title('Estimate of A - Laplace noise');xlabel('Sample size (N)');ylabel('Mean Square Error (MSE)')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Autore: Fulvio Gini 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: fulvio.gini@unipi.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 