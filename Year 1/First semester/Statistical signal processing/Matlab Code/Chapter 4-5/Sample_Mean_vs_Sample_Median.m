%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola la stima del parametro A (valor medio e
% mediana dei dati) mediante gli stimatori Sample Mean e Sample Median
% per rumore Gaussiano oppure di Laplace
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=10^5;   % Numero di prove dell'esperimento
n=[1:N]';

A=2;                % valor medio v.a. Gaussiana/Laplace
varx=1;             % varianza v.a. Gaussiana (G)
eta=sqrt(varx/2);   % parametro della v.a. di Laplace (L) in modo che G ed L abbiano la stessa potenza

uno=ones(1,N);

X=sqrt(varx)*randn(N,1)+A;   % Si generano N campioni di una v.a. Gaussiana
U=rand(N,1);
XE=-eta*log(U);              % Si generano N campioni di una v.a. Exp. Monolatera E{XE}=eta
S=sign(rand(N,1)-0.5);
XL=S.*XE+A;                  % Si generano N campioni di una v.a. di Laplace

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h=waitbar(0,'Please wait...');

for k=1:N
    waitbar(k/N,h); 
    z(k)=mean(X(1:k));
    y(k)=median(X(1:k));
    ze(k)=mean(XL(1:k));
    ye(k)=median(XL(1:k));
end

 close(h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1)
semilogx(n,z,'blue', 'LineWidth',2);grid on;
hold on
semilogx(n,y,'red', 'LineWidth',2);
legend('Sample Mean','Sample Median');
title('Estimate of A - Gaussian noise');xlabel('Sample size (N)');ylabel('Sample Mean vs Sample Median')

figure (2);
semilogx(n,ze,'blue', 'LineWidth',2);grid on;
hold on
semilogx(n,ye,'red', 'LineWidth',2);
legend('Sample Mean','Sample Median');
title('Estimate of A - Laplace noise');xlabel('Sample size (N)');ylabel('Sample Mean vs Sample Median')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Autore: Fulvio Gini 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: fulvio.gini@unipi.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 